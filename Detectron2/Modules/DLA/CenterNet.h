#pragma 
#include <Detectron2/Structures/Box2BoxTransform.h>
#include <Detectron2/Structures/ImageList.h>
#include <Detectron2/Structures/Instances.h>
#include <Detectron2/Structures/Matcher.h>
#include <Detectron2/Structures/ShapeSpec.h>
#include <Detectron2/Modules/RPN/StandardRPNHead.h>
#include <Detectron2/Modules/RPN/RPN.h>
#include <Detectron2/Modules/RPN/AnchorGenerator.h>
#include <Detectron2/Modules/DLA/CenterNetHead.h>
namespace Detectron2
{
	class CenterNetImpl : public torch::nn::Module {
	public:

		CenterNetImpl(CfgNode &cfg, const ShapeSpec::Map& input_shape);

		void initialize(const ModelImporter& importer, const std::string& prefix);
		std::tuple<torch::Tensor, torch::Tensor> heatmap_focal_loss(torch::Tensor inputs,
			torch::Tensor targets,
			torch::Tensor pos_inds,
			torch::Tensor labels,
			float alpha = -1,
			float beta = 4,
			float gamma = 2,
			std::string reduction = "sum",
			float sigmoid_clamp = 1e-4,
			float ignore_high_fp = -1);

		std::tuple<torch::Tensor, torch::Tensor> binary_heatmap_focal_loss(
			torch::Tensor inputs,
			torch::Tensor targets,
			torch::Tensor pos_inds,
			float alpha = -1,
			float beta = 4,
			float gamma = 2,
			float sigmoid_clamp = 1e-4,
			float ignore_high_fp = -1.);

		std::tuple<InstancesList, TensorMap> inference(ImageList images, std::vector<torch::Tensor> clss_per_level,
			std::vector<torch::Tensor> reg_pred_per_level, std::vector<torch::Tensor> agn_hm_pred_per_level,
			std::vector<torch::Tensor> grids);

		InstancesPtr ml_nms(InstancesPtr boxlist, float nms_thresh, int max_proposals = -1);

		InstancesList nms_and_topK(InstancesList boxlists, bool nms = true);

		std::tuple<torch::Tensor, torch::Tensor> _add_more_pos(torch::Tensor reg_pred,
			InstancesList gt_instances, std::vector<torch::Tensor> shapes_per_level);

		InstancesList predict_single_level(torch::Tensor grids, torch::Tensor heatmap,
			torch::Tensor reg_pred, std::vector<ImageSize> image_sizes, torch::Tensor agn_hm, int level,
			bool is_proposal= false);

		InstancesList predict_instances(std::vector<torch::Tensor> grids, std::vector<torch::Tensor> logits_pred,
			std::vector<torch::Tensor> reg_pred, std::vector<ImageSize> image_sizes, std::vector<torch::Tensor> agn_hm_pred,
			bool is_proposal = false);

		torch::Tensor iou_loss(torch::Tensor pred, torch::Tensor target,
			torch::Tensor weight = {}, std::string reduction = "sum");
		TensorMap losses(torch::Tensor pos_inds, torch::Tensor labels, torch::Tensor reg_targets, torch::Tensor flattened_hms,
			torch::Tensor logits_pred, torch::Tensor reg_pred, torch::Tensor agn_hm_pred);
		std::tuple<InstancesList, TensorMap>forward(const ImageList& images, const TensorMap& features_,
			const InstancesList& gt_instances = {});
		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> _flatten_outputs(std::vector<torch::Tensor> clss, std::vector<torch::Tensor> reg_pred, std::vector<torch::Tensor> agn_hm_pred);
	private:
		std::vector<std::string> m_in_features;

		AnchorGenerator m_anchor_generator{ nullptr };
		std::shared_ptr<Box2BoxTransform> m_box2box_transform;
		Matcher m_anchor_matcher;
		RPNHead m_rpn_head{ nullptr };

		int post_nms_topk_train;
		int post_nms_topk_test;
		std::string loc_loss_type;
		int min_radius;
		float delta;
		float score_thresh;
		float nms_thresh_train;
		float nms_thresh_test;
		int num_classes;
		int pre_nms_topk_train;
		int pre_nms_topk_test;
		bool not_nms;
		bool center_nms;
		bool with_agn_hm;
		bool as_proposal;
		bool only_proposal;
		bool not_norm_reg;
		float sigmoid_clamp;
		float ignore_high_fp;
		float loss_gamma;
		float reg_weight;
		float hm_focal_alpha;
		float hm_focal_beta;
		float pos_weight;
		float neg_weight;
		std::vector<std::vector<int>> size_of_interest;
		bool more_pos;
		int m_min_box_side_len;
		float m_nms_thresh;
		int m_batch_size_per_image;
		float m_positive_fraction;
		float m_smooth_l1_beta;
		float m_loss_weight;
		int m_pre_nms_topk[2];
		int m_post_nms_topk[2];
		int m_boundary_threshold;
		std::vector<float> self_strides;
		CenterNetHead m_centernet_head;

		std::vector<torch::Tensor> _transpose(std::vector<torch::Tensor> training_targets, std::vector<int> num_loc_list);
		torch::Tensor _create_heatmaps_from_dist(torch::Tensor dist,
			torch::Tensor labels, int channels);
		torch::Tensor _create_agn_heatmaps_from_dist(torch::Tensor dist);
		torch::Tensor _get_reg_targets(torch::Tensor reg_targets, torch::Tensor dist,
			torch::Tensor mask, torch::Tensor area);
		torch::Tensor assign_reg_fpn(torch::Tensor reg_targets_per_im, torch::Tensor size_ranges);
		torch::Tensor assign_fpn_level(torch::Tensor bboxes);
		std::vector<torch::Tensor> compute_grids(TensorVec features);
		torch::Tensor get_center3x3(torch::Tensor m_grids, torch::Tensor centers, torch::Tensor strides);

		std::tuple<torch::Tensor, torch::Tensor> _get_label_inds(const InstancesList& gt_instances, std::vector<torch::Tensor> shapes_per_level);

		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> _get_ground_truth(std::vector<torch::Tensor> grids,
			std::vector<torch::Tensor> shapes_per_level, const InstancesList& gt_instances);
	};
	TORCH_MODULE(CenterNet);
}