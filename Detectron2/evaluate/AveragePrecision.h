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
	void inc_good_prediction(int value = 1) { tp += value; }
	void inc_bad_prediction(int value = 1) { fp += value; }
	void inc_not_prediction(int value = 1) { fn += value; }

	int TP() { return tp;}
	int FP() { return fp; }
	int FN() { return fn; }

private:
	int tp = 0;
	int fp = 0;
	int fn = 0;
};

float Precision(int TP, int FP, int FN) {
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

float Recall(int TP, int FP, int FN) {
	int total_gt = TP + FN;
	if (total_gt == 0) {
		return 1;
	}
	return float(TP) / total_gt;
}

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

	return inter / union_;
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


class Accumulators {
public:
	Accumulators(int num_classes, float pr_scale, float overlap_threshold = 0.25) {
		m_num_classes = num_classes;
		m_pr_scale = pr_scale;
		m_overlap_threshold = overlap_threshold;
		total_accumulators.resize(m_num_classes);
	}

	void reset_accumulators() {
		for (int j = 0; j < m_num_classes; j++) {
			total_accumulators.push_back(APAccumulator());
		}
	}

	torch::Tensor compute_true_positive(torch::Tensor mask) {
		return torch::sum(mask.any(0));
	}

	torch::Tensor compute_IoU_mask(torch::Tensor pred, torch::Tensor target, float confidence_threshold) {
		auto iou = jaccard(pred, target);

		for (int i = 0; i < pred.sizes()[0]; i++) {
			auto maxj = iou.index({ i, Slice() }).argmax().item().toInt();
			iou.index({ i, Slice(0, maxj, 1) }) = 0;
			iou.index({ i, Slice(maxj + 1, iou.sizes()[1], 1) }) = 0;
		}
		auto indexes = iou >= m_overlap_threshold;
		return indexes;
	}
	std::vector<APAccumulator> evaluate_(torch::Tensor IOUmask, std::vector<APAccumulator> accumulators, torch::Tensor pred_classes, torch::Tensor pred_conf,
		torch::Tensor gt_classes, float confidence_threshold) {
		int count = 0;
		for (auto& iter : accumulators) {
			auto gt_number = torch::sum(gt_classes.data() == count);
			auto pred_mask = (pred_classes.data() == 0 & pred_conf.data() >= confidence_threshold);
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

	std::vector<APAccumulator> evaluate(torch::Tensor pred_bb, torch::Tensor pred_classes, torch::Tensor pred_conf, torch::Tensor gt_bb, torch::Tensor gt_classes) {
		torch::Tensor IOUmask = {};
		if (pred_bb.sizes()[0] > 0) {
			IOUmask = compute_IoU_mask(pred_bb, gt_bb, m_overlap_threshold);
		}

		total_accumulators = evaluate_(IOUmask, total_accumulators, pred_classes, pred_conf, gt_classes, m_pr_scale);
		return total_accumulators;
	}

private:
	int m_num_classes;
	float m_pr_scale;
	float m_overlap_threshold;
	std::vector<APAccumulator> total_accumulators;
};
