#pragma once
#include <math.h>
#include <array>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <torch/torch.h>

using namespace std;
using namespace torch;
using namespace torch::indexing;

class APAccumulator {
public:
	APAccumulator(int tp = 0, int fp = 0, int fn = 0) {
		TP = tp;
		FP = fp;
		FN = fn;
	};
	void inc_good_prediction(int value = 1) { TP += value;}
	void inc_bad_prediction(int value = 1) { FP += value; }
	void inc_not_prediction(int value = 1) { FN += value; }

	float precision() {
		int total_predicted = TP + FP;
		if (total_predicted == 0) {
			int total_gt = TP + FN;
			if (total_gt == 0) {
				return 1;
			}
			else {
				return 0;
			}
		}
		return float(TP) / total_predicted;
	}

	float recall() {
		int total_gt = TP + FN;
		if (total_gt == 0) {
			return 1;
		}
		return float(TP) / total_gt;
	}

	string __str__() {
		string str = "";
		str += "True positives: " + to_string(TP) + "\n";
		str += "False positives: " + to_string(FP) + "\n";
		str += "False Negatives: " + to_string(FN) + "\n";
		str += "Precision: " + to_string(precision()) + "\n";
		str += "Recall: " + to_string(recall()) + "\n";

		return str;
	}
private:
	int TP = 0;
	int FP = 0;
	int FN = 0;
};

torch::Tensor intersect_area(torch::Tensor box_a, torch::Tensor box_b) {
	auto resized_a = box_a.view({ box_a.sizes()[0],1,box_a.sizes()[1] });
	auto resized_b = box_b.view({ 1,box_b.sizes()[0],box_b.sizes()[1] });
	auto max_xy = torch::minimum(resized_a.index({ Slice(), Slice(), Slice(2,4,1) }), resized_b.index({ Slice(), Slice(), Slice(2,4,1) }));
	auto min_xy = torch::maximum(resized_a.index({ Slice(), Slice(), Slice(0,2,1) }), resized_b.index({ Slice(), Slice(), Slice(0,2,1) }));

	auto diff_xy = max_xy - min_xy;
	auto inter = torch::clip(diff_xy, 0, torch::max(diff_xy).item().toFloat());

	return inter.index({ Slice(),Slice(),0 }) * inter.index({ Slice(),Slice(),1 });
}

torch::Tensor jaccard(torch::Tensor box_a, torch::Tensor box_b) {
	auto inter = intersect_area(box_a, box_b);
	auto area_a = (box_a.index({ Slice(),2 }) - box_a.index({ Slice(),0 })) * (box_a.index({ Slice(),3 }) - box_a.index({ Slice(),1 }));
	auto area_b = (box_b.index({ Slice(),2 }) - box_b.index({ Slice(),0 })) * (box_b.index({ Slice(),3 }) - box_b.index({ Slice(),1 }));
	area_a = area_a.view({ area_a.sizes()[0],1 });
	area_b = area_b.view({ 1,area_b.sizes()[0] });
	auto union_ = area_a + area_b - inter;

	return inter/union_;
}

class MAP {
public:
	MAP(int num_classes, int pr_samples = 11, float overlap_threshold = 0.5) {
		m_num_classes = num_classes;
		m_pr_samples = pr_samples;
		m_overlap_threshold = overlap_threshold;
		pr_scale = torch::linspace(0, 1, m_pr_samples);
		total_accumulators.resize(m_pr_samples);
		for (int i = 0; i < m_pr_samples; i++) {
			total_accumulators[i].resize(m_num_classes);
		}
	}

	void reset_accumulators() {
		for (int i = 0; i < m_pr_samples; i++) {
			std::vector<APAccumulator> class_accumulators;
			for (int j = 0; j < m_num_classes; j++) {
				class_accumulators.push_back(APAccumulator());
			}
			total_accumulators.push_back(class_accumulators);
		}
	}

	torch::Tensor compute_true_positive(torch::Tensor mask) {
		return torch::sum(mask.any(0));
	}

	float compute_ap(std::vector<float> precisions, std::vector<float> recalls) {
		float previous_recall = 0;
		float average_precision = 0;
		for (int i = precisions.size() - 1; i > 0; i--) {
			auto precision = precisions[i];
			auto recall = recalls[i];
			average_precision += precision * (recall - previous_recall);
			previous_recall = recall;
		}
		return average_precision;
	}

	struct pr_recall {
		std::vector<float> precisions;
		std::vector<float> recalls;
	};

	pr_recall compute_precision_recall_(int class_index, bool interpolated = true) {
		std::vector<float> precisions;
		std::vector<float> recalls;
		precisions.reserve(m_pr_samples);
		recalls.reserve(m_pr_samples);

		for (auto& iter : total_accumulators) {
			precisions.push_back(iter[class_index].precision());
			recalls.push_back(iter[class_index].recall());
		}

		if (interpolated) {
			std::vector<float> interpolated_precision;
			for (auto& precision : precisions) {
				float last_max = 0;
				if (interpolated_precision.size() != 0) {
					auto position = *max_element(interpolated_precision.begin(), interpolated_precision.end());
					last_max = interpolated_precision[position-1];
				}
				interpolated_precision.push_back(std::max(precision, last_max));
			}
			precisions = interpolated_precision;
		}
		pr_recall res;
		res.precisions = precisions;
		res.recalls = recalls;
		return res;
	}

	torch::Tensor compute_IoU_mask(torch::Tensor pred, torch::Tensor target, float confidence_threshold) {
		auto iou = jaccard(pred, target);

		for (int i = 0; i < pred.sizes()[0]; i++) {
			auto maxj = iou.index({ i, Slice() }).argmax().item().toInt();
			iou.index({ i, Slice(0, maxj, 1) }) = 0;
			iou.index({ i, Slice(maxj+1, iou.sizes()[1], 1) }) = 0;
		}
		auto indexes = iou >= m_overlap_threshold;
		return indexes;
	}
	std::vector<APAccumulator> evaluate_(torch::Tensor IOUmask, std::vector<APAccumulator> accumulators, torch::Tensor pred_classes, torch::Tensor pred_conf,
		torch::Tensor gt_classes, torch::Tensor confidence_threshold) {
		auto count = torch::zeros(1);
		for (auto& iter : accumulators) {
			auto gt_number = torch::sum(gt_classes == count);
			auto pred_mask = (pred_classes == count & pred_conf >= confidence_threshold );
			auto pred_number = torch::sum(pred_mask);
			if (pred_number.item().toInt() == 0) {
				iter.inc_not_prediction(gt_number.item().toInt());
				continue;
			}
			//std::cout << "iter: " << iter << std::endl;
			torch::Tensor iou1, mask;
			iou1 = IOUmask.index({ pred_mask,Slice() });
			mask = iou1.index({ Slice(),gt_classes == count });

			int tp = compute_true_positive(mask).item().toInt();
			int fp = pred_number.item().toInt() - tp;
			int fn = gt_number.item().toInt() - tp;
			iter.inc_good_prediction(tp);
			iter.inc_bad_prediction(fp);
			iter.inc_not_prediction(fn);
			count += 1;
		}
		return accumulators;
	}

	void evaluate(torch::Tensor pred_bb, torch::Tensor pred_classes, torch::Tensor pred_conf, torch::Tensor gt_bb, torch::Tensor gt_classes) {
		if (pred_bb.dim() == 1) {
			pred_bb = torch::repeat_interleave(pred_bb.expand({ pred_bb.sizes()[0],1 }), 4, 1);
		}
		torch::Tensor IOUmask = {};
		if (pred_bb.sizes()[0] > 0) {
			IOUmask = compute_IoU_mask(pred_bb, gt_bb, m_overlap_threshold);
		}

		std::cout<<total_accumulators.size() << std::endl;
		for (int i = 0; i < pr_scale.sizes()[0]; i++) {
			auto accumulators = total_accumulators[i];
			auto r = pr_scale.index({i });
			total_accumulators[i] = evaluate_(IOUmask, accumulators, pred_classes, pred_conf, gt_classes, r);
		}
	}

	float mAP(bool interpolated = true, std::vector<string> names = {}) {
		std::vector<float> mean_average_precision = {};
		float total_AP = 0;
		for (int i = 0; i < m_num_classes; i++) {
			auto pr = compute_precision_recall_(i, interpolated);
			auto average_precision = compute_ap(pr.precisions, pr.recalls);
			total_AP += average_precision;
			mean_average_precision.push_back(average_precision);
		}
		return total_AP / m_num_classes;
	}

private:
	int m_num_classes;
	int m_pr_samples;
	float m_overlap_threshold;
	torch::Tensor pr_scale;
	std::vector<std::vector<APAccumulator>> total_accumulators;
};