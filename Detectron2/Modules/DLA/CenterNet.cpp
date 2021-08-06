#include <assert.h>
#include "Base.h"
#include "CenterNet.h"
#include <Detectron2/Structures/NMS.h>
#include <Detectron2/Structures/Sampling.h>
#include <Detectron2/Modules/RPN/RPNOutputs.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double INF = 100000000;

CenterNetImpl::CenterNetImpl(CfgNode& cfg, const ShapeSpec::Map& input_shapes) :
	m_in_features(cfg["MODEL.CENTERNET.IN_FEATURES"].as<vector<string>>()),
	m_anchor_matcher(
		cfg["MODEL.RPN.IOU_THRESHOLDS"].as<vector<float>>(),
		cfg["MODEL.RPN.IOU_LABELS"].as<vector<int>>(),
		true // allow_low_quality_matches
	),
	m_min_box_side_len(cfg["MODEL.PROPOSAL_GENERATOR.MIN_SIZE"].as<int>()),
	m_nms_thresh(cfg["MODEL.RPN.NMS_THRESH"].as<float>()),
	m_batch_size_per_image(cfg["MODEL.RPN.BATCH_SIZE_PER_IMAGE"].as<int>()),
	m_positive_fraction(cfg["MODEL.RPN.POSITIVE_FRACTION"].as<float>()),
	m_smooth_l1_beta(cfg["MODEL.RPN.SMOOTH_L1_BETA"].as<float>()),
	m_loss_weight(cfg["MODEL.RPN.LOSS_WEIGHT"].as<float>()),
	m_pre_nms_topk{ // Map from m_training state to train / test settings
	/* false : */ cfg["MODEL.RPN.PRE_NMS_TOPK_TEST"].as<int>(),
	/* true: */ cfg["MODEL.RPN.PRE_NMS_TOPK_TRAIN"].as<int>()
	},
	m_post_nms_topk{
	/* false : */ cfg["MODEL.RPN.POST_NMS_TOPK_TEST"].as<int>(),
	/* true: */ cfg["MODEL.RPN.POST_NMS_TOPK_TRAIN"].as<int>()
	},
	m_boundary_threshold(cfg["MODEL.RPN.BOUNDARY_THRESH"].as<int>()),
	self_strides(cfg["MODEL.CENTERNET.FPN_STRIDES"].as<vector<float>>())
{
	num_classes = cfg["MODEL.CENTERNET.NUM_CLASSES"].as<int>();
	score_thresh = cfg["MODEL.CENTERNET.INFERENCE_TH"].as<float>();
	min_radius = cfg["MODEL.CENTERNET.MIN_RADIUS"].as<int>();
	hm_focal_alpha = cfg["MODEL.CENTERNET.HM_FOCAL_ALPHA"].as<float>();
	hm_focal_beta = cfg["MODEL.CENTERNET.HM_FOCAL_BETA"].as<float>();
	loss_gamma = cfg["MODEL.CENTERNET.LOSS_GAMMA"].as<float>();
	reg_weight = cfg["MODEL.CENTERNET.REG_WEIGHT"].as<float>();
	not_norm_reg = cfg["MODEL.CENTERNET.NOT_NORM_REG"].as<bool>();
	with_agn_hm = cfg["MODEL.CENTERNET.WITH_AGN_HM"].as<bool>();
	only_proposal = cfg["MODEL.CENTERNET.ONLY_PROPOSAL"].as<bool>();
	as_proposal = cfg["MODEL.CENTERNET.AS_PROPOSAL"].as<bool>();
	not_nms = cfg["MODEL.CENTERNET.NOT_NMS"].as<bool>();
	pos_weight = cfg["MODEL.CENTERNET.POS_WEIGHT"].as<float>();
	neg_weight = cfg["MODEL.CENTERNET.NEG_WEIGHT"].as<float>();
	sigmoid_clamp = cfg["MODEL.CENTERNET.SIGMOID_CLAMP"].as<float>();
	ignore_high_fp = cfg["MODEL.CENTERNET.IGNORE_HIGH_FP"].as<float>();
	center_nms = cfg["MODEL.CENTERNET.CENTER_NMS"].as<bool>();
	size_of_interest = cfg["MODEL.CENTERNET.SOI"].as<vector<vector<int>>>();
	more_pos = cfg["MODEL.CENTERNET.MORE_POS"].as<bool>();
	auto more_pos_thresh = cfg["MODEL.CENTERNET.MORE_POS_THRESH"].as<float>();
	auto more_pos_topk = cfg["MODEL.CENTERNET.MORE_POS_TOPK"].as<int>();
	pre_nms_topk_train = cfg["MODEL.CENTERNET.PRE_NMS_TOPK_TRAIN"].as<int>();
	pre_nms_topk_test = cfg["MODEL.CENTERNET.PRE_NMS_TOPK_TEST"].as<int>();
	post_nms_topk_train = cfg["MODEL.CENTERNET.POST_NMS_TOPK_TRAIN"].as<int>();
	post_nms_topk_test = cfg["MODEL.CENTERNET.POST_NMS_TOPK_TEST"].as<int>();
	nms_thresh_train = cfg["MODEL.CENTERNET.NMS_TH_TRAIN"].as<float>();
	nms_thresh_test = cfg["MODEL.CENTERNET.NMS_TH_TEST"].as<float>();
	//auto debug = cfg["DEBUG"].as<bool>();
	auto device = cfg["MODEL.DEVICE"].as<string>();
	auto pixel_mean = cfg["MODEL.PIXEL_MEAN"].as<vector<float>>();
	auto pixel_std = cfg["MODEL.PIXEL_STD"].as<vector<float>>();
	//auto vis_thresh = cfg["VIS_THRESH"].as<float>();
	if (center_nms) {
		not_nms = true;
	}
	delta = (1 - cfg["MODEL.CENTERNET.HM_MIN_OVERLAP"].as<float>()) / (1 + cfg["MODEL.CENTERNET.HM_MIN_OVERLAP"].as<float>());
	loc_loss_type = cfg["MODEL.CENTERNET.LOC_LOSS_TYPE"].as<std::string>();
	/*auto iou_loss = IOULOSS(loc_loss_type);
	assert(not only_proposal || with_agn_hm);
	auto hm_min_overlap = cfg["MODEL.CENTERNET.HM_MIN_OVERLAP"].as<float>();
	auto delta = (1 - hm_min_overlap) / (1 + hm_min_overlap);
	auto centernet_head = CenterNetHead(cfg, input_shape_head);
	
	if (debug) {
		pixel_mean = torch::from_blob(pixel_mean, {1,3}).to(device).view({ 3,1,1 });
		pixel_std = torch::from_blob(pixel_std, {1,3}).to(device).view({ 3,1,1 });
	}*/

	m_box2box_transform = Box2BoxTransform::Create(cfg["MODEL.RPN.BBOX_REG_WEIGHTS"]);
	
	auto filtered = ShapeSpec::filter(input_shapes, m_in_features);
	m_centernet_head = CenterNetHead(cfg, input_shapes);
	register_module("centernet_head", m_centernet_head);
	//m_anchor_generator = build_anchor_generator(cfg, filtered);
	//register_module("anchor_generator", m_anchor_generator);
	//m_rpn_head = build_rpn_head(cfg, filtered);
	//register_module("rpn_head", m_rpn_head);
}

void CenterNetImpl::initialize(const ModelImporter& importer, const std::string& prefix) {
	//m_anchor_generator->initialize(importer, prefix + ".anchor_generator");
	//m_rpn_head->initialize(importer, prefix + ".rpn_head");
}

std::vector<torch::Tensor> CenterNetImpl::compute_grids(TensorVec features)
{
	auto device = features[0].device();
	std::vector<torch::Tensor> grids;
	for (int level = 0; level< features.size(); level++)
	{
		auto m_features = features[level];
		auto size = m_features.sizes();
		auto h = size[2];
		auto w = size[3];
		auto shifts_x = torch::arange(0, w * self_strides[level], self_strides[level], torch::kFloat32).to(device);
		auto shifts_y = torch::arange(0, h * self_strides[level], self_strides[level], torch::kFloat32).to(device);
		auto mesh_shifts = torch::meshgrid({ shifts_y, shifts_x });
		mesh_shifts[0] = mesh_shifts[0].reshape(-1);
		mesh_shifts[1] = mesh_shifts[1].reshape(-1);
		auto grids_per_level = torch::stack({ mesh_shifts[1] ,mesh_shifts[0]},1) + self_strides[level] / 2;
		grids.push_back(grids_per_level);
	}
	return grids;
}

torch::Tensor CenterNetImpl::assign_fpn_level(torch::Tensor bboxes)
{
	vector<torch::Tensor> size_ranges_ve;
	for (int i = 0 ; i< size_of_interest.size();i++)
	{
		size_ranges_ve.push_back(torch::tensor(size_of_interest[i]));
	}
	auto size_ranges = torch::stack(size_ranges_ve);
	//cout << "bboxes" << bboxes << endl;
	auto boxes_2 = bboxes.index({ Slice(), Slice(2,4,1) });
	//cout << "boxes_2" << boxes_2 << endl;
	auto boxes_1 = bboxes.index({ Slice(), Slice(0,2,1) });
	//cout << "boxes_1" << boxes_1 << endl;

	auto crit = sqrt(((boxes_2 - boxes_1) * (boxes_2 - boxes_1)).sum(1)) / 2;
	int n = crit.sizes()[0];
	int L = size_ranges.sizes()[0];
	crit = crit.view({ n,1 }).expand({n,L});
	auto size_ranges_expand = size_ranges.view({ 1, L, 2 }).expand({ n, L, 2 }).to(crit.device());
	//std::cout << crit.device() << std::endl;
	//std::cout << size_ranges_expand.device() << std::endl;
	auto is_cared_in_the_level = (crit >= size_ranges_expand.index({ Slice(),Slice(), 0 }))
		& (crit <= size_ranges_expand.index({ Slice(),Slice(), 1 }));

	//cout << "is_cared_in_the_level" << is_cared_in_the_level << endl;
	return is_cared_in_the_level;
}

std::tuple<torch::Tensor, torch::Tensor> CenterNetImpl::_get_label_inds(const InstancesList& gt_instances, std::vector<torch::Tensor> shapes_per_level)
{
	auto device = shapes_per_level[0].device();
	int L = self_strides.size();
	int B = gt_instances.size();
	for (auto m_shapes_per_level : shapes_per_level)
	{
		m_shapes_per_level.toType(torch::kLong);
	}
	auto m_shapes_per_level = torch::stack(shapes_per_level);
	//cout << "m_shapes_per_level" << m_shapes_per_level << endl;
	auto loc_per_level = m_shapes_per_level.index({ Slice(), 0 }) * m_shapes_per_level.index({ Slice(), 1 });
	//cout << "m_shapes_per_level" << loc_per_level << endl;
	vector<torch::Tensor> level_bases_ve;
	torch::Tensor s = torch::zeros(1).toType(torch::kLong).to(device);
	for (int i = 0; i < L; i++)
	{
		level_bases_ve.push_back(s);
		s = s + B * loc_per_level[i];
	}
	auto level_bases = torch::stack(level_bases_ve).to(device);
	//cout << "level_bases:" << level_bases << endl;
	auto strides_default = torch::tensor(self_strides).to(device);
	//cout << "strides_default:" << strides_default << endl;
	std::vector<torch::Tensor> pos_inds_ve;
	std::vector<torch::Tensor> labels_ve;
	for (int im_i = 0; im_i < B; im_i++)
	{
		auto targets_per_im = gt_instances[im_i];
		auto bboxes = targets_per_im->getTensor("gt_boxes");
		//cout << "bboxes:" << bboxes << endl;
		int n = bboxes.sizes()[0];
		auto c = bboxes.index({ Slice(), Slice(0,2,1) }).to(device);
		//cout << "c" << c << endl;
		auto centers = (bboxes.index({ Slice(), Slice(0,2,1) }) + bboxes.index({ Slice(), Slice(2,4,1) }).to(device)) / 2;
		//cout << "centers:" << centers << endl;
		centers = centers.view({ n, 1, 2 }).expand({ n, L, 2 }).to(device);
		//cout << "centers:" << centers << endl;
		auto strides = strides_default.view({ 1, L, 1 }).expand({ n, L, 2 }).to(device);
		//cout << "strides:" << strides << endl;
		auto centers_inds = (centers / strides).toType(torch::kLong);
		//cout << "centers_inds:" << centers_inds << endl;

 		auto Ws = m_shapes_per_level.index({ Slice(), 1 }).view({ 1, L }).expand({ n, L }).to(device);
		//cout << "Ws:" << Ws << endl;

		auto centes_1 = centers_inds.index({ Slice(),Slice(), 1 }).to(device);
		//cout << "centes_1:" << centes_1 << endl;
		auto centes_0 = centers_inds.index({ Slice(),Slice(), 0 }).to(device);//Slice(0,1,1)
		//cout << "centes_0:" << centes_0 << endl;

		auto pos_ind = level_bases.view({ 1, L }).expand({ n, L })
			+ im_i * loc_per_level.view({ 1, L }).expand({ n, L })
			+ centers_inds.index({ Slice(),Slice(), 1 }).to(device) * Ws
			+ centers_inds.index({ Slice(),Slice(), 0 }).to(device);
		auto is_cared_in_the_level = assign_fpn_level(bboxes);
		//cout << "is_cared_in_the_level" << is_cared_in_the_level << endl;
		pos_ind = pos_ind.index({ is_cared_in_the_level }).to(device);
		//cout << "pos_ind:" << pos_ind << endl;
		auto label = targets_per_im->getTensor("gt_classes").view(
			{ n, 1 }).expand({ n, L }).index({is_cared_in_the_level}).view(-1);
		//cout << "label:" << label << endl;
		int a = 0;
		pos_inds_ve.push_back(pos_ind);
		labels_ve.push_back(label);
	}
	auto pos_inds = torch::cat(pos_inds_ve, 0).toType(kLong);
	auto labels = torch::cat(labels_ve, 0).toType(kLong);

	//cout << "pos_inds:" << pos_inds << endl;
	//cout << "labels:" << labels << endl;
	return { pos_inds ,labels };
}

torch::Tensor CenterNetImpl::get_center3x3(torch::Tensor locations, torch::Tensor centers, torch::Tensor strides)
{

	int M = locations.sizes()[0];
	int N = centers.sizes()[0];
	auto locations_expanded = locations.view({ M, 1, 2 }).expand({ M, N, 2 });
	//cout << "locations_expanded" << locations_expanded.sizes() << endl;
	auto centers_expanded = centers.view({ 1, N, 2 }).expand({ M, N, 2 });
	auto strides_expanded = strides.view({ M, 1, 1 }).expand({ M, N, 2 });
	auto centers_discret = ((centers_expanded / strides_expanded).toType(torch::kInt) *
		strides_expanded).toType(torch::kFloat) + strides_expanded / 2;
	//cout << "centers_discret" << centers_discret.sizes() << endl;
	auto dist_x = (locations_expanded.index({ Slice(),Slice(), 0 })
		- centers_discret.index({ Slice(),Slice(), 0 })).abs();
	//cout << "dist_x" << dist_x.sizes() << endl;
	auto dist_y = (locations_expanded.index({ Slice(),Slice(), 1 })
		- centers_discret.index({ Slice(),Slice(), 1 })).abs();
	torch::Tensor mm = (dist_x <= strides_expanded.index({ Slice(),Slice(), 0 })) &
	(dist_y <= strides_expanded.index({ Slice(),Slice(), 0 }));
	//cout << "mm" << mm.sizes() << endl;
	return mm;
}

torch::Tensor CenterNetImpl::assign_reg_fpn(torch::Tensor reg_targets_per_im, torch::Tensor size_ranges)
{
	auto device = reg_targets_per_im.device();
	auto boxes_2 = reg_targets_per_im.index({ Slice(), Slice(),Slice(0,2,1) });
	auto boxes_1 = reg_targets_per_im.index({ Slice(),Slice(), Slice(2,4,1) });
	auto crit = sqrt(((boxes_2 + boxes_1) * (boxes_2 + boxes_1)).sum(2)) / 2;
	//cout << "crit" << crit << endl;

	auto is_cared_in_the_level = (crit >= size_ranges.index({ Slice(),Slice(0,1,1) }).to(device))
		& (crit <= size_ranges.index({ Slice(), Slice(1,2,1) }).to(device));
	//cout << "is_cared_in_the_level" << is_cared_in_the_level << endl;
	return is_cared_in_the_level;
}

torch::Tensor CenterNetImpl::_get_reg_targets(torch::Tensor reg_targets, torch::Tensor dist,
	torch::Tensor mask,torch::Tensor area)
{
	torch::Tensor min_dist, min_inds;
	dist.index_put_({ mask == 0 }, INF * 1.0);
	tie(min_dist, min_inds) = dist.min(1);
	auto reg_targets_per_im = reg_targets.index({ Slice(),0,  Slice() });
	std::vector<torch::Tensor> reg_targets_per_im_ve;
	int num_size = reg_targets.sizes()[1];
	for (int i = 0; i < num_size; i++)
	{
		auto l_reg_targets_per_im = reg_targets.index({ Slice(),i,  Slice() });
		reg_targets_per_im_ve.push_back(l_reg_targets_per_im);
	}

	for (int j = 0; j < reg_targets_per_im.sizes()[0]; j++)
	{
		if (min_inds[j].item().toInt() != 0)
		{
			reg_targets_per_im[j] = reg_targets_per_im_ve[min_inds[j].item().toInt()][j];
		}
	}

	reg_targets_per_im.index_put_({ min_dist == INF }, -INF);
	return reg_targets_per_im.to(reg_targets.device());
}

torch::Tensor CenterNetImpl::_create_agn_heatmaps_from_dist(torch::Tensor dist)
{
	auto heatmaps = torch::zeros({ dist.sizes()[0], 1});
	auto ex = torch::exp(-get<0>(dist.min(1)));
	//std::cout << "ex" << ex << endl;
	heatmaps.index_put_({ Slice() ,0}, torch::exp(-get<0>(dist.min(1))));
	auto zeros = heatmaps < 1e-4;
	heatmaps.index_put_({ zeros },0);
	return heatmaps;
}

torch::Tensor CenterNetImpl::_create_heatmaps_from_dist(torch::Tensor dist,
	torch::Tensor labels, int channels)
{
	auto heatmaps = torch::zeros({ dist.sizes()[0], channels });
	for (int c = 0; c < channels; c++)
	{
		auto inds = (labels == c);
		//if (inds.toType(torch::kInt).sum() == 0)
		//{
		//	continue;
		//}
		heatmaps.index_put_({ Slice() ,c }, torch::exp(-get<0>(dist.index({ Slice() ,inds }).min(1))));
		auto zeros = heatmaps.index({ Slice() ,c }) < 1e-4;
		heatmaps.index_put_({ zeros ,c }, 0);
	}
	return heatmaps;
}

vector<torch::Tensor> CenterNetImpl::_transpose(vector<torch::Tensor> training_targets,vector<int> num_loc_list) //?
{
	vector<torch::Tensor> m_training_targets_0;
	vector<torch::Tensor> m_training_targets_1;
	vector<torch::Tensor> m_training_targets_2;
	for (int im_i = 0; im_i < training_targets.size(); im_i++)
	{
		vector<torch::Tensor> m_ve = torch::split_with_sizes(training_targets[im_i], { num_loc_list[0],num_loc_list[1],num_loc_list[2] }, 0);
		m_training_targets_0.push_back(m_ve[0]);
		m_training_targets_1.push_back(m_ve[1]);
		m_training_targets_2.push_back(m_ve[2]);
	}
	vector<torch::Tensor> targets_level_first;
	targets_level_first.push_back(torch::cat(m_training_targets_0, 0));
	targets_level_first.push_back(torch::cat(m_training_targets_1, 0));
	targets_level_first.push_back(torch::cat(m_training_targets_2, 0));
	return targets_level_first;
}

std::tuple<torch::Tensor, torch::Tensor,torch::Tensor, torch::Tensor> CenterNetImpl::_get_ground_truth(std::vector<torch::Tensor> grids,
	std::vector<torch::Tensor> shapes_per_level, const InstancesList& gt_instances)
{
	auto device = grids[0].device();
	torch::Tensor pos_inds;
	torch::Tensor labels;
	if (!more_pos)
	{
		tie(pos_inds, labels) = _get_label_inds(gt_instances, shapes_per_level);
	}
	//cout << "pos_inds:" << pos_inds << endl;
	auto heatmap_channels = num_classes;
	auto L = grids.size();
	vector<int> num_loc_list;
	for (auto m_grids : grids)
	{
		num_loc_list.push_back(m_grids.sizes()[0]);
	}
	vector<torch::Tensor> strides_ve;
	for (int l = 0; l < L; l++)
	{
		strides_ve.push_back(torch::ones(num_loc_list[l]) * self_strides[l]);
	}
	auto strides = torch::cat(strides_ve).to(device);
	//cout << "strides:" << strides << endl;

	vector<torch::Tensor> reg_size_ranges_ve;
	for (int l = 0; l < L; l++)
	{
		reg_size_ranges_ve.push_back(torch::tensor(size_of_interest[l]).toType(torch::kFloat)
			.view({ 1, 2 }).expand({ num_loc_list[l], 2 }));
	}
	auto reg_size_ranges = torch::cat(reg_size_ranges_ve);
	//cout << "reg_size_ranges:" << reg_size_ranges << endl;

	//cout << "grids:" << grids << endl;
	auto m_grids = torch::cat(grids,0);
	int M = m_grids.sizes()[0];

	vector<torch::Tensor> reg_targets;
	vector<torch::Tensor> flattened_hms;
	for (int i = 0; i < gt_instances.size(); i++)
	{
		auto boxes = gt_instances[i]->getTensor("gt_boxes");
		//cout << "boxes:" << boxes << endl;
		auto boxes_x1 = boxes.index({ Slice(), Slice(0,1,1) });
		auto boxes_x2 = boxes.index({ Slice(), Slice(2,3,1) });
		auto boxes_y1 = boxes.index({ Slice(), Slice(1,2,1) });
		auto boxes_y2 = boxes.index({ Slice(), Slice(3,4,1) });
		auto area = ((boxes_x2 - boxes_x1) * (boxes_y2 - boxes_y1));
		//cout << "area:" << area << endl;
		auto gt_classes = gt_instances[i]->getTensor("gt_classes");
		int N = boxes.sizes()[0];
		if (N == 0)
		{
			reg_targets.push_back(torch::zeros({ M, 4 }) - INF);
			int _channels;
			if (only_proposal)
			{
				_channels = 1;
			}
			else
			{
				_channels = heatmap_channels;
			}
			flattened_hms.push_back(torch::zeros({ M, _channels }));
		}
		auto m_grids_1 =   m_grids.index({ Slice(), Slice(0,1,1) });
		//cout << "m_grids_1:" << m_grids_1 << endl;

		//cout << "boxes:" << boxes << endl;
		auto boxes_1 = boxes.index({ Slice(), Slice(0,1,1) });
		//cout << "boxes_1:" << boxes_1 << endl;

		auto l = m_grids.index({ Slice(), Slice(0,1,1) }).view({M,1}) - 
			boxes.index({ Slice(), Slice(0,1,1) }).view({ 1,N });
		//cout << "l:" << l << endl;
		auto t = m_grids.index({ Slice(), Slice(1,2,1) }).view({ M,1 }) -
			boxes.index({ Slice(), Slice(1,2,1) }).view({ 1,N });
		//cout << "t:" << t << endl;
		auto r = boxes.index({ Slice(), Slice(2,3,1) }).view({ 1,N }) -
			m_grids.index({ Slice(), Slice(0,1,1) }).view({ M,1 });
		//cout << "r:" << r << endl;
		auto b = boxes.index({ Slice(), Slice(3,4,1) }).view({ 1,N }) -
			m_grids.index({ Slice(), Slice(1,2,1) }).view({ M,1 });
		//cout << "b:" << b << endl;
		auto reg_target = torch::stack({ l, t, r, b }, 2).to(device);
		//cout << "reg_target:" << reg_target << endl;

		auto centers = (boxes.index({ Slice(), Slice(0,2,1) }) + boxes.index({ Slice(), Slice(2,4,1) })) / 2;
		//cout << "centers:" << centers << endl;
		auto centers_expanded = centers.view({ 1, N, 2 }).expand({ M, N, 2 }).to(device);
		//cout << "centers_expanded:" << centers_expanded << endl;
		auto strides_expanded = strides.view({ M, 1, 1 }).expand({ M, N, 2 }).to(device);
		//cout << "strides_expanded:" << strides_expanded << endl;
		auto centers_discret = ((centers_expanded / strides_expanded).toType(torch::kInt)
			* strides_expanded).toType(torch::kFloat) + strides_expanded /2;
		//cout << "centers_discret:" << centers_discret << endl;
		auto is_peak = pow((m_grids.view({ M, 1, 2 }).expand({ M, N, 2 }) - centers_discret),2).sum(2) == 0;
		//cout << "is_peak:" << is_peak << endl;
		//auto m_reg_target = reg_target.min(2);
		//auto m_reg_target_0 = get<0>(m_reg_target);
		auto is_in_boxes = get<0>(reg_target.min(2)) > 0;
		//cout << "is_in_boxes:" << is_in_boxes << endl;
		auto is_center3x3_m = get_center3x3(m_grids, centers, strides);
		//cout << "is_center3x3_m:" << is_center3x3_m << endl;
		auto is_center3x3 = is_center3x3_m & is_in_boxes;
		//cout << "is_center3x3" << is_center3x3 << endl;
		auto is_cared_in_the_level = assign_reg_fpn(reg_target, reg_size_ranges);
		auto reg_mask = is_center3x3 & is_cared_in_the_level;
		auto dist2 = (pow((m_grids.view({ M, 1, 2 }).expand({ M, N, 2 }) - \
			centers_expanded),2)).sum(2);
		dist2.index({ is_peak }) = 0;
		auto radius2 = pow(delta,2) * 2 * area;
		radius2 = torch::clamp(radius2, pow(min_radius, 2));
		auto weighted_dist2 = dist2 / radius2.view({ 1, N }).expand({ M, N });
		//cout << "reg_target" << reg_target << endl;
		//cout << "weighted_dist2" << weighted_dist2 << endl;
		//cout << "reg_mask" << reg_mask << endl;
		//cout << "area" << area << endl;

		//cout << "reg_target" << reg_target << endl;
		reg_target = _get_reg_targets(reg_target, weighted_dist2.clone(), reg_mask, area);
		//cout << "reg_target" << reg_target << endl;

		torch::Tensor flattened_hm;
		if (only_proposal)
		{
			flattened_hm = _create_agn_heatmaps_from_dist(weighted_dist2.clone());
		}
		else
		{
			flattened_hm = _create_heatmaps_from_dist(
				weighted_dist2.clone(), gt_classes, heatmap_channels);
		}
		//cout << "flattened_hm" << flattened_hm << endl;
		reg_targets.push_back(reg_target);
		//cout << "reg_target:" << reg_target << endl;
		flattened_hms.push_back(flattened_hm);
	}
	//cout << "reg_targets:" << reg_targets << endl;
	//cout << "flattened_hms:" << flattened_hms << endl;
	auto m_reg_targets = _transpose(reg_targets, num_loc_list);
	auto m_flattened_hms = _transpose(flattened_hms, num_loc_list);
	for (int l = 0; l < m_reg_targets.size(); l++)
	{
		m_reg_targets[l] = m_reg_targets[l] / self_strides[l];
	}
	auto reg_targets_result  = cat(m_reg_targets, 0);
	//cout << "reg_targets_result" << reg_targets_result << endl;
	auto flattened_hms_result = cat(m_flattened_hms, 0);
	//cout << "flattened_hms_result" << flattened_hms_result << endl;
	return { pos_inds, labels, reg_targets_result, flattened_hms_result };
	//int a = 0;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CenterNetImpl::_flatten_outputs(vector<torch::Tensor> clss, vector<torch::Tensor> reg_pred, vector<torch::Tensor> agn_hm_pred)
{
	torch::Tensor m_clss, m_reg_pred, m_agn_hm_pred;
	if (clss.size() != 0)
	{
		for (auto &x : clss)
		{
			//cout << "x" << x << endl;
			x = x.permute({ 0, 2, 3, 1 }).reshape({ -1, x.sizes()[1] });
			//cout << "x" << x << endl;
		}
		m_clss = cat(clss,0);
	}

	for (auto &x : reg_pred)
	{
		//cout << "x" << x << endl;
		x = x.permute({ 0, 2, 3, 1 }).reshape({ -1, 4 });
		//cout << "x" << x << endl;
	}
	//cout << "reg_pred:" << reg_pred << endl;
	m_reg_pred = cat(reg_pred, 0);

	if (with_agn_hm)
	{
		for (auto &x : agn_hm_pred)
		{
			//cout << "x" << x << endl;
			x = x.permute({ 0, 2, 3, 1 }).reshape(-1);
			//cout << "x" << x << endl;
		}
		m_agn_hm_pred = cat(agn_hm_pred, 0);
	}
	return { m_clss, m_reg_pred, m_agn_hm_pred };
}

std::tuple<torch::Tensor, torch::Tensor> CenterNetImpl::heatmap_focal_loss(torch::Tensor inputs,
	torch::Tensor targets,
	torch::Tensor pos_inds,
	torch::Tensor labels,
	float alpha,
	float beta,
	float gamma,
	string reduction,
	float sigmoid_clamp,
	float ignore_high_fp)
{
	auto pred = torch::clamp(inputs.sigmoid_(), sigmoid_clamp, 1 - sigmoid_clamp);
	auto neg_weights = torch::pow(1 - targets, beta);
	auto pos_pred_pix = pred.index({pos_inds});
	auto pos_pred = pos_pred_pix.gather(1, labels.unsqueeze(1));
	auto pos_loss = torch::log(pos_pred) * torch::pow(1 - pos_pred, gamma);
	auto neg_loss = torch::log(1 - pred) * torch::pow(pred, gamma) * neg_weights;

	if (ignore_high_fp > 0)
	{
		auto not_high_fp = (pred < ignore_high_fp).toType(torch::kFloat);
		neg_loss = not_high_fp * neg_loss;
	}
	if (strcmp(reduction.c_str(), "sum") == 0)
	{
		pos_loss = pos_loss.sum();
		neg_loss = neg_loss.sum();
	}

	if (alpha >= 0)
	{
		pos_loss = alpha * pos_loss;
		neg_loss = (1 - alpha) * neg_loss;
	}
	return { -pos_loss, -neg_loss };
}

torch::Tensor CenterNetImpl::iou_loss(torch::Tensor pred, torch::Tensor target,
	torch::Tensor weight, string reduction)
{
	//cout << "pred:" << pred << endl;
	//cout << "target:" << target << endl;
	//cout << "weight:" << weight << endl;
	torch::Tensor losses;
	auto pred_left = pred.index({ Slice(),0 });
	auto pred_top = pred.index({ Slice(),1 });
	auto pred_right = pred.index({ Slice(),2 });
	auto pred_bottom = pred.index({ Slice(),3 });

	auto target_left = target.index({ Slice(),0 });
	auto target_top = target.index({ Slice(),1 });
	auto target_right = target.index({ Slice(),2 });
	auto target_bottom = target.index({ Slice(),3 });

	auto target_aera = (target_left + target_right) *
		(target_top + target_bottom);
	auto pred_aera = (pred_left + pred_right) * 
		(pred_top + pred_bottom);

	auto w_intersect = torch::min(pred_left, target_left) + 
		torch::min(pred_right, target_right);
	auto h_intersect = torch::min(pred_bottom, target_bottom) +
		torch::min(pred_top, target_top);

	auto g_w_intersect = torch::max(pred_left, target_left) + 
		torch::max(pred_right, target_right);
	auto g_h_intersect = torch::max(pred_bottom, target_bottom) + \
		torch::max(pred_top, target_top);

	auto ac_uion = g_w_intersect * g_h_intersect;
	auto area_intersect = w_intersect * h_intersect;
	auto area_union = target_aera + pred_aera - area_intersect;

	auto ious = (area_intersect + 1.0) / (area_union + 1.0);
	auto gious = ious - (ac_uion - area_union) / ac_uion;

	//cout << "ious" << ious << endl;
	//cout << "gious" << gious << endl;

	if (strcmp(loc_loss_type.c_str(),"iou") == 0)
	{
		losses = -torch::log(ious);
	}
	else if (strcmp(loc_loss_type.c_str(), "linear_iou") == 0)
	{
		losses = 1 - ious;
	}
	else if (strcmp(loc_loss_type.c_str(), "giou") == 0)
	{
		losses = 1 - gious;
	}

	if (weight.numel() != 0)
	{
		losses = losses * weight;
	}
	else
	{
		losses = losses;
	}

	torch::Tensor sum;
	if (strcmp(reduction.c_str(), "sum") == 0)
	{
		sum = losses.sum();
	}
	else if (strcmp(reduction.c_str(), "batch") == 0)
	{
		sum = losses.sum(1);
	}
	else if(strcmp(reduction.c_str(), "none") == 0)
	{
		sum = losses;
	}
	return sum;
}

std::tuple<torch::Tensor, torch::Tensor> CenterNetImpl::binary_heatmap_focal_loss(
	torch::Tensor inputs,
	torch::Tensor targets,
	torch::Tensor pos_inds,
	float alpha,
	float beta,
	float gamma,
	float sigmoid_clamp,
	float ignore_high_fp)
{
	auto pred = torch::clamp(inputs.sigmoid_(), sigmoid_clamp, 1 - sigmoid_clamp);
	auto neg_weights = torch::pow(1 - targets, beta);
	auto pos_pred = pred.index({ pos_inds });
	auto pos_loss = torch::log(pos_pred) * torch::pow(1 - pos_pred, gamma);
	auto neg_loss = torch::log(1 - pred) * torch::pow(pred, gamma) * neg_weights;
	if (ignore_high_fp > 0)
	{
		auto not_high_fp = (pred < ignore_high_fp).toType(torch::kFloat);
		neg_loss = not_high_fp * neg_loss;
	}
	pos_loss = -pos_loss.sum();
	neg_loss = -neg_loss.sum();

	if (alpha >= 0)
	{
		pos_loss = alpha * pos_loss;
		neg_loss = (1 - alpha) * neg_loss;
	}
	return { pos_loss, neg_loss };
}

TensorMap CenterNetImpl::losses(torch::Tensor pos_inds, torch::Tensor labels, torch::Tensor reg_targets, torch::Tensor flattened_hms,
	torch::Tensor logits_pred, torch::Tensor reg_pred, torch::Tensor agn_hm_pred)
{
	//cout << "pos_inds:" << pos_inds << endl;
	//cout << "labels:" << labels << endl;
	//cout << "reg_targets:" << reg_targets << endl;
	//cout << "flattened_hms:" << flattened_hms << endl;
	//cout << "logits_pred:" << logits_pred << endl;
	//cout << "reg_pred:" << reg_pred << endl;
	//cout << "agn_hm_pred:" << agn_hm_pred << endl;
	TensorMap loss;
	int num_pos_local = pos_inds.numel();
	auto num_gpus = 1; //?
	auto total_num_pos = num_pos_local;//?
	auto num_pos_avg = MAX(total_num_pos / num_gpus, 1.0);

	if (!only_proposal)
	{
		torch::Tensor pos_loss, neg_loss;
		tie(pos_loss, neg_loss) = heatmap_focal_loss(
			logits_pred, flattened_hms, pos_inds, labels,
			hm_focal_alpha,
			hm_focal_beta,
			loss_gamma,
			"sum",
			sigmoid_clamp,
			ignore_high_fp
		);
		pos_loss = pos_weight * pos_loss / num_pos_avg;
		neg_loss = neg_weight * neg_loss / num_pos_avg;
		loss["loss_centernet_pos"] = pos_loss;
		loss["loss_centernet_neg"] = neg_loss;
	}
	auto reg_inds = torch::nonzero(get<0>(reg_targets.max(1)) >= 0).squeeze(1);
	reg_pred = reg_pred.index({ reg_inds });
	auto reg_targets_pos = reg_targets.index({ reg_inds });
	auto reg_weight_map = get<0>(flattened_hms.max(1));
	reg_weight_map = reg_weight_map.index({ reg_inds });
	if (not_norm_reg)
	{
		reg_weight_map = reg_weight_map * 0 + 1;
	}
	//cout << "reg_weight_map.sum()" << reg_weight_map.sum().item().toInt() << endl;
	auto reg_norm = MAX(reg_weight_map.sum().item().toInt() / num_gpus, 1.0);
	//cout << "reg_weight:" << reg_weight << endl;
	//cout << "reg_pred:" << reg_pred << endl;
	//cout << "reg_targets_pos:" << reg_targets_pos << endl;
	//cout << "reg_weight_map:" << reg_weight_map << endl;
	auto reg_loss = reg_weight * iou_loss(
		reg_pred, reg_targets_pos, reg_weight_map, "sum") / reg_norm;
	loss["loss_centernet_loc"] = reg_loss;
	//cout << "loss_centernet_loc" << reg_loss << endl;
	if (with_agn_hm)
	{
		auto cat_agn_heatmap =get<0>(flattened_hms.max(1));
		torch::Tensor agn_pos_loss, agn_neg_loss;
		tie(agn_pos_loss, agn_neg_loss) = binary_heatmap_focal_loss(
			agn_hm_pred, cat_agn_heatmap, pos_inds,
			hm_focal_alpha,
			hm_focal_beta,
			loss_gamma,
			sigmoid_clamp,
			ignore_high_fp);
		agn_pos_loss = pos_weight * agn_pos_loss / num_pos_avg;
		agn_neg_loss = neg_weight * agn_neg_loss / num_pos_avg;
		loss["loss_centernet_agn_pos"] = agn_pos_loss;
		loss["loss_centernet_agn_neg"] = agn_neg_loss;
		//cout << "loss_centernet_agn_pos" << agn_pos_loss << endl;
		//cout << "loss_centernet_agn_neg" << agn_neg_loss << endl;
	}
	return loss;
}

std::tuple<torch::Tensor, torch::Tensor> CenterNetImpl::_add_more_pos(torch::Tensor reg_pred, InstancesList gt_instances, std::vector<torch::Tensor> shapes_per_level)
{
	torch::Tensor pos_inds;
	torch::Tensor labels;

	return { pos_inds ,labels };
}

InstancesList CenterNetImpl::predict_single_level(torch::Tensor grids, torch::Tensor heatmap,
	torch::Tensor reg_pred, std::vector<ImageSize> image_sizes, torch::Tensor agn_hm,int level,
	bool is_proposal)
{
	InstancesList results;
	int N = heatmap.sizes()[0];
	int C = heatmap.sizes()[1];
	int H = heatmap.sizes()[2];
	int W = heatmap.sizes()[3];
	if (center_nms)
	{
		//torch::nn::functional::MaxPool2dFuncOptions options(1);
		//options.kernel_size();
		//auto heatmap_nms = torch::nn::functional::max_pool2d(
		//	heatmap, options.stride(2).padding(0));
		auto heatmap_nms = max_pool2d(heatmap, 3, 1, 1);
		heatmap = heatmap * (heatmap_nms == heatmap).toType(torch::kFloat);
	}
	heatmap = heatmap.permute({ 0, 2, 3, 1 });
	heatmap = heatmap.reshape({ N, -1, C });
	auto box_regression = reg_pred.view({ N, 4, H, W }).permute({ 0, 2, 3, 1 });
	box_regression = box_regression.reshape({ N, -1, 4 });
	auto candidate_inds = heatmap > score_thresh;
	auto pre_nms_top_n = candidate_inds.view({ N, -1 }).sum(1);
	int pre_nms_topk;
	if (is_training())
	{
		pre_nms_topk = pre_nms_topk_train;
	}
	else
	{
		pre_nms_topk = pre_nms_topk_test;
	}
	pre_nms_top_n = pre_nms_top_n.clamp(pre_nms_topk);

	if (agn_hm.numel() != 0 )
	{
		agn_hm = agn_hm.view({ N, 1, H, W }).permute({ 0, 2, 3, 1 });
		agn_hm = agn_hm.reshape({ N, -1 });
		heatmap = heatmap * agn_hm.index({ Slice(),Slice(), None });
	}

	for (int i = 0; i < N; i++)
	{
		auto per_box_cls = heatmap[i];
		auto per_candidate_inds = candidate_inds[i];
		per_box_cls = per_box_cls.index({ per_candidate_inds });

		auto per_candidate_nonzeros = per_candidate_inds.nonzero();
		auto per_box_loc = per_candidate_nonzeros.index({ Slice(), 0 });
		auto per_class = per_candidate_nonzeros.index({ Slice(), 1 });

		auto per_box_regression = box_regression[i];
		per_box_regression = per_box_regression.index({ per_box_loc });
		auto per_grids = grids.index({ per_box_loc });

		auto per_pre_nms_top_n = pre_nms_top_n[i];

		if (per_candidate_inds.sum().item().toInt() > per_pre_nms_top_n.item().toInt())
		{
			torch::Tensor top_k_indices;
			tie(per_box_cls, top_k_indices) = per_box_cls.topk(per_pre_nms_top_n.item().toInt(), false);
			per_class = per_class.index({ top_k_indices });
			per_box_regression = per_box_regression.index({ top_k_indices });
			per_grids = per_grids.index({ top_k_indices });
		}
		auto detections = torch::stack({
			per_grids.index({ Slice(), 0 }) - per_box_regression.index({ Slice(), 0 }),
				per_grids.index({ Slice(), 1 }) - per_box_regression.index({ Slice(), 1 }),
				per_grids.index({ Slice(), 0 }) + per_box_regression.index({ Slice(), 2 }),
				per_grids.index({ Slice(), 1 }) + per_box_regression.index({ Slice(), 3 }),
		}, 1);
		detections.index({ Slice(), 2 }) = torch::max(detections.index({ Slice(), 2 }), 
			detections.index({ Slice(), 0 }) + 0.01);
		detections.index({ Slice(), 3 }) = torch::max(detections.index({ Slice(), 3 }), 
			detections.index({ Slice(), 1 }) + 0.01);
		auto boxlist = make_shared<Instances>(image_sizes[i]);
		if (with_agn_hm)
		{
			auto m_per_box_cls = torch::sqrt(per_box_cls);
			boxlist->set("scores", m_per_box_cls);
		}
		else
		{
			boxlist->set("scores", per_box_cls);
		}
		boxlist->set("pred_boxes", detections);
		boxlist->set("pred_classes", per_class);
		results.push_back(boxlist);
	}
	return results;
}

InstancesPtr CenterNetImpl::ml_nms(InstancesPtr boxlist, float nms_thresh, int max_proposals)
{
	if (nms_thresh <= 0)
	{
		return boxlist;
	}

	torch::Tensor boxes, labels;
	if (boxlist->has("pred_boxes"))
	{
		boxes = boxlist->getTensor("pred_boxes");
		labels = boxlist->getTensor("pred_classes");
	}
	else
	{
		boxes = boxlist->getTensor("proposal_boxes");
		labels = torch::zeros(boxes.numel());
	}
	auto scores = boxlist->getTensor("scores");
	auto keep = Detectron2::batched_nms(boxes, scores, labels, nms_thresh);
	if (max_proposals > 0)
	{
		keep = keep.index({ Slice(),max_proposals });
	}
	auto m_boxlist = boxlist->index({ keep });
	return dynamic_pointer_cast<Instances>(m_boxlist);
}

InstancesList CenterNetImpl::nms_and_topK(InstancesList boxlists, bool nms)
{
	InstancesList results;
	int num_images = boxlists.size();
	for (int i = 0; i < num_images; i++)
	{
		float nms_thresh;
		if (is_training())
		{
			nms_thresh = nms_thresh_train;
		}
		else
		{
			nms_thresh = nms_thresh_test;
		}
		InstancesPtr result;
		if (nms)
		{
			result = ml_nms(boxlists[i], nms_thresh);
		}
		else
		{
			result = boxlists[i];
		}
		int num_dets = result->len();
		int post_nms_topk;
		if (is_training())
		{
			post_nms_topk = post_nms_topk_train;
		}
		else
		{
			post_nms_topk = post_nms_topk_test;
		}

		auto tensor_boxes = result->getTensor("pred_boxes");

		std::cout << "num_dets: " << num_dets << " post_nms_topk: " << post_nms_topk << std::endl;
		if (num_dets > post_nms_topk)
		{
			SequencePtr result_1;
			auto cls_scores = result->getTensor("scores");
			torch::Tensor image_thresh, _;
			tie(image_thresh, _) = torch::kthvalue(
				cls_scores.cpu(),
				num_dets - post_nms_topk + 1
			);
			//cout << "image_thresh:" << image_thresh << endl;
			//cout << "_:" << _ << endl;
			//cout << "cls_scores:" << cls_scores << endl;
			auto keep = cls_scores >= image_thresh.item();
			//cout << "keep" << keep << endl;
			keep = torch::nonzero(keep).squeeze(1);
			if (keep.numel() != post_nms_topk)
			{
				keep = keep.index({ Slice(0,post_nms_topk,1) });
			}
			cout << "keep_num:" << keep.numel() << endl;
			result_1 = result->index({ keep });
			auto tensor_boxes = dynamic_pointer_cast<Instances>(result_1)->getTensor("pred_boxes");
			results.push_back(dynamic_pointer_cast<Instances>(result_1));
		}
		else
		{
			results.push_back(dynamic_pointer_cast<Instances>(result));
		}

	}
	return results;
}

InstancesList CenterNetImpl::predict_instances(std::vector<torch::Tensor> grids, std::vector<torch::Tensor> logits_pred,
	std::vector<torch::Tensor> reg_pred, std::vector<ImageSize> image_sizes, std::vector<torch::Tensor> agn_hm_pred,
	bool is_proposal)
{
	vector<InstancesList> sampled_boxes;
	for (int l = 0; l < grids.size(); l++)
	{
		torch::Tensor m_agn_hm_pred;
		if (agn_hm_pred.size() == 0)
		{
			sampled_boxes.push_back(predict_single_level(
				grids[l], logits_pred[l], reg_pred[l] * self_strides[l],
				image_sizes, m_agn_hm_pred, l, is_proposal));
		}
		else
		{
			sampled_boxes.push_back(predict_single_level(
				grids[l], logits_pred[l], reg_pred[l] * self_strides[l],
				image_sizes, agn_hm_pred[l], l, is_proposal));
		}
	}
	InstancesList boxlists;
	for (int i = 0; i < sampled_boxes[0].size(); i++)
	{
		if (grids.size() == sampled_boxes.size())
		{
			vector<torch::Tensor> sampled_boxes_scores_ve;
			vector<torch::Tensor> sampled_boxes_boxes_ve;
			vector<torch::Tensor> sampled_boxes_classes_ve;
			for (int j = 0; j < sampled_boxes.size(); j++)
			{
				auto sampled_boxes_0 = sampled_boxes[j][i];
				auto sampled_boxes_scores = sampled_boxes_0->getTensor("scores");
				sampled_boxes_scores_ve.push_back(sampled_boxes_scores);
				auto sampled_boxes_boxes = sampled_boxes_0->getTensor("pred_boxes");
				sampled_boxes_boxes_ve.push_back(sampled_boxes_boxes);
				auto sampled_boxes_classes = sampled_boxes_0->getTensor("pred_classes");
				sampled_boxes_classes_ve.push_back(sampled_boxes_classes);
			}
			auto tensor_scores = torch::cat(sampled_boxes_scores_ve);
			auto tensor_boxes = torch::cat(sampled_boxes_boxes_ve);
			auto tensor_classes = torch::cat(sampled_boxes_classes_ve);
			auto boxlist = make_shared<Instances>(image_sizes[i]);
			boxlist->set("scores", tensor_scores);
			boxlist->set("pred_boxes", tensor_boxes);
			boxlist->set("pred_classes", tensor_classes);
			boxlists.push_back(boxlist);
		}
	}
	cout<<"boxlists.scores.size"<< boxlists[0]->getTensor("scores").sizes();
	boxlists = nms_and_topK(boxlists, !not_nms);
	return boxlists;
}

std::tuple<InstancesList, TensorMap> CenterNetImpl::inference(ImageList images, std::vector<torch::Tensor> clss_per_level,
	std::vector<torch::Tensor> reg_pred_per_level, std::vector<torch::Tensor> agn_hm_pred_per_level,
	std::vector<torch::Tensor> grids)
{
	for (auto &x : clss_per_level)
	{
		if (x.numel() != 0)
		{
			x = x.sigmoid();
		}
	}

	for (auto& x : agn_hm_pred_per_level)
	{
		if (x.numel() != 0)
		{
			x = x.sigmoid();
		}
	}

	InstancesList proposals;
	if (only_proposal)
	{
		proposals = predict_instances(
			grids, agn_hm_pred_per_level, reg_pred_per_level,
			images.image_sizes(), {});
	}
	else
	{
		proposals = predict_instances(
			grids, clss_per_level, reg_pred_per_level,
			images.image_sizes(), agn_hm_pred_per_level);
	}
	if (only_proposal || as_proposal)
	{
		for (int p = 0; p < proposals.size(); p++)
		{
			auto tensor_boxes = proposals[p]->getTensor("pred_boxes");
			proposals[p]->set("proposal_boxes", tensor_boxes);
			auto tensor_scores = proposals[p]->getTensor("scores");
			//cout << "tensor_scores" << tensor_scores << endl;
			proposals[p]->set("objectness_logits", tensor_scores);
			proposals[p]->remove("pred_boxes");
		}
	}
	return { proposals, {} };
}

std::tuple<InstancesList, TensorMap> CenterNetImpl::forward(const ImageList& images, const TensorMap& features_,
	const InstancesList& gt_instances) {
	TensorVec features;
	for (auto f : m_in_features) {
		auto iter = features_.find(f);
		assert(iter != features_.end());
		features.push_back(iter->second);
	}
	auto centernet = m_centernet_head->forward(features);
	auto clss_per_level = centernet.clss;
	auto reg_pred_per_level = centernet.bbox_reg;
	auto agn_hm_pred_per_level = centernet.agn_hms;
	auto grids = compute_grids(features);
	//cout << "grids:" << grids << endl;

	std::vector<torch::Tensor> shapes_per_level;
	for (int i = 0; i < centernet.bbox_reg.size(); i++)
	{
		auto size = centernet.bbox_reg[i].sizes();
		auto device = centernet.bbox_reg[i].device();
		shapes_per_level.push_back(torch::tensor({ size[2],size[3] }).to(device));
	}
	//cout << "shapes_per_level:" << shapes_per_level << endl;
	InstancesList proposals;
	TensorMap losses_;
	if (is_training()) {
		torch::Tensor pos_inds, labels, reg_targets, flattened_hms;
		tie(pos_inds, labels, reg_targets, flattened_hms) = _get_ground_truth(grids, shapes_per_level, gt_instances);
		torch::Tensor logits_pred, reg_pred, agn_hm_pred;
		//cout << "clss_per_level:" << clss_per_level << endl;
		//cout << "reg_pred_per_level:" << reg_pred_per_level << endl;
		//cout << "agn_hm_pred_per_level:" << agn_hm_pred_per_level << endl;
		tie(logits_pred, reg_pred, agn_hm_pred) = _flatten_outputs(clss_per_level, reg_pred_per_level, agn_hm_pred_per_level);
		//cout << "logits_pred:" << logits_pred << endl;
		//cout << "reg_pred:" << reg_pred << endl;
		//cout << "agn_hm_pred:" << agn_hm_pred << endl;
		if (more_pos)
		{
			tie(pos_inds, labels)= _add_more_pos(
				reg_pred, gt_instances, shapes_per_level);
		}
		auto device = flattened_hms.device();
		//std::cout << device << std::endl;
		losses_ = losses(pos_inds.to(device), labels.to(device), reg_targets.to(device), flattened_hms,
			logits_pred, reg_pred.to(device), agn_hm_pred.to(device));
		/*losses_ = losses(pos_inds, labels, reg_targets, flattened_hms,
			logits_pred, reg_pred, agn_hm_pred);*/


		if (only_proposal)
		{
			for (auto& x: agn_hm_pred_per_level)
			{
				x = x.sigmoid();
			}
			proposals = predict_instances(
				grids, agn_hm_pred_per_level, reg_pred_per_level,
				gt_instances.getImageSizes(), clss_per_level);
		}
		else if (as_proposal)
		{
			for (auto& x : clss_per_level)
			{
				x = x.sigmoid();
			}
			proposals = predict_instances(
				grids, clss_per_level, reg_pred_per_level,
				gt_instances.getImageSizes(), agn_hm_pred_per_level);
		}
		if (only_proposal || as_proposal)
		{
			for (int p = 0; p < proposals.size(); p++)
			{
				auto tensor_boxes = proposals[p]->getTensor("pred_boxes");
				proposals[p]->set("proposal_boxes", tensor_boxes);
				auto tensor_scores = proposals[p]->getTensor("scores");
				proposals[p]->set("objectness_logits", tensor_scores);
				proposals[p]->remove("pred_boxes");
				proposals[p]->remove("scores");
				proposals[p]->remove("pred_classes");
			}
		}
	}
	else
	{
		return inference(
			images, clss_per_level, reg_pred_per_level,
			agn_hm_pred_per_level, grids);
	}

	return { proposals,losses_ };
}