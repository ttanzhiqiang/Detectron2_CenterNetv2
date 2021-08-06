#include <assert.h>
#include <string>
#include "Base.h"
#include <Detectron2/Modules/Conv/ConvBn2d.h>
#include <Detectron2/Modules/Conv/DeformConv.h>
#include <Detectron2/Modules/BatchNorm/BatchNorm.h>
#include "CenterNetHead.h"
#include<math.h>
#include<memory>
#include<map>
#include<unordered_map>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CenterNetHeadImpl::CenterNetHeadImpl(CfgNode& cfg, const ShapeSpec::Map& input_shapes) {
	m_with_agn_hm = cfg["MODEL.CENTERNET.WITH_AGN_HM"].as<bool>();
	m_only_proposal = cfg["MODEL.CENTERNET.ONLY_PROPOSAL"].as<bool>();
	auto num_classes = cfg["MODEL.CENTERNET.NUM_CLASSES"].as<int64_t>();
	auto in_features = cfg["MODEL.CENTERNET.IN_FEATURES"].as<vector<string>>();
	int out_kernel = 3;
	auto norm = cfg["MODEL.CENTERNET.NORM"].as<string>();
	auto prior_prob = cfg["MODEL.CENTERNET.PRIOR_PROB"].as<float>();
	m_bias_value = -log((1 - prior_prob) / prior_prob);

	convs_deform m_convs_deform_cls;
	m_convs_deform_cls.num_convs = m_only_proposal ? 0 : cfg["MODEL.CENTERNET.NUM_CLS_CONVS"].as<int64_t>();
	m_convs_deform_cls.use_deformable = cfg["MODEL.CENTERNET.USE_DEFORMABLE"].as<bool>();
	m_head_configs["cls"] = m_convs_deform_cls;
	convs_deform m_convs_deform_bbox;
	m_convs_deform_bbox.num_convs = cfg["MODEL.CENTERNET.NUM_BOX_CONVS"].as<int64_t>();
	//m_convs_deform_bbox.num_convs = 1;
	m_convs_deform_bbox.use_deformable = cfg["MODEL.CENTERNET.USE_DEFORMABLE"].as<bool>();
	m_head_configs["bbox"] = m_convs_deform_bbox;
	convs_deform m_convs_deform_share;
	m_convs_deform_share.num_convs = cfg["MODEL.CENTERNET.NUM_SHARE_CONVS"].as<int64_t>();
	m_convs_deform_share.use_deformable = cfg["MODEL.CENTERNET.USE_DEFORMABLE"].as<bool>();
	m_head_configs["share"] = m_convs_deform_share;

	std::vector<int> in_channels;
	in_channels.reserve(in_features.size());
	for (auto& f : in_features) {
		auto iter= input_shapes.find(f);
		assert(iter != input_shapes.end());
		auto& item = iter->second;
		in_channels.push_back(item.channels);
	}

	auto in_channel = in_channels[0];

	std::unordered_map<string, int> channels = { {"cls",in_channel} ,{"bbox",in_channel},{"share",in_channel} };
	//m_bbox_tower = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, in_channel, 3).stride(1).padding(1).bias(true));
	//register_module("conv", m_bbox_tower);

	//m_bbox_tower = ConvBn2d(nn::Conv2dOptions(in_channel, in_channel, 3).stride(1).padding(1).bias(true));
	//register_module("conv", m_bbox_tower);

	for (auto iter = m_head_configs.begin(); iter != m_head_configs.end(); iter++) {
		auto tower = nn::Sequential();
		string name = iter->first;
		auto tower_name = "m_" + name + "_tower";
		m_tower_names.push_back(tower_name);
		auto convs_deform = iter->second;
		auto num_convs = convs_deform.num_convs;
		auto use_deformable = convs_deform.use_deformable;
		auto channel = channels[name];
		for (int i = 0; i < num_convs; i++) {
			if (use_deformable && i == num_convs - 1) {
				tower->push_back(DeformConv(i == 0 ? in_channel : channel, channel, 3, 1, 1, 0, 32, 0, true, BatchNorm::Type::kGN));
			}
			else {
				tower->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(i == 0 ? in_channel : channel, channel, 3).stride(1).padding(1).bias(true)));
			}
			if (strcmp(norm.c_str(), "GN") && channel % 32 != 0) {
				tower->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(25, channel)));  
			}
			else if (strcmp(norm.c_str(), " "))  {
				tower->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, channel)));
			}
			tower->push_back(torch::nn::ReLU());
		}
		m_tower[tower_name] = tower;
		//std::cout << tower << std::endl;
		//if (strcmp(tower_name.c_str(), "m_bbox_tower") == 0) {
		//	for (int i = 0; i < tower.get()->size();i++) {
		//		auto a = tower->children()[i];
		//		m_bbox_tower->push_back(tower->children()[i]);
		//	}
		//	m_bbox_tower->push_back(tower);
		//}
		register_module(tower_name, tower);
	}
	m_bbox_pred = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, 4, out_kernel).stride(1).padding(out_kernel / 2));
	register_module("bbox_pred", m_bbox_pred);

	for (int i = 0; i < input_shapes.size(); i++) {
		auto m_scale = Scale(1.0);
		m_scales->push_back(m_scale);	
	}
	register_module("scales", m_scales);

	if (m_with_agn_hm) {
		m_agn_hm = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, 1, out_kernel).stride(1).padding(out_kernel / 2));
		register_module("agn_hm", m_agn_hm);
	}

	if (m_only_proposal == false) {
		m_cls_logits = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, num_classes, out_kernel).stride(1).padding(out_kernel / 2));
		register_module("cls_logits", m_cls_logits);

	}
}

void CenterNetHeadImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	for (std::string& tower_name:m_tower_names) {
		for (auto& iter : m_tower[tower_name]->children()) {
			if (strcmp(iter->name().c_str(), "Conv2d")) {
				auto conv = iter->as<torch::nn::Conv2d>();
				torch::nn::init::normal_(conv->weight, 0.01);
				torch::nn::init::constant_(conv->bias, 0);
			}
		}
	}

	torch::nn::init::constant_(m_bbox_pred->bias, 8.);

	if (m_with_agn_hm) {
		torch::nn::init::constant_(m_agn_hm->bias, m_bias_value);
		torch::nn::init::normal_(m_agn_hm->weight, 0.01);
	}

	if (m_only_proposal == false) {
		torch::nn::init::constant_(m_cls_logits->bias, m_bias_value);
		torch::nn::init::normal_(m_cls_logits->weight, 0.01);
	}
}

CenterNetVec CenterNetHeadImpl::forward(TensorVec x) {
	CenterNetVec ret;
	std::vector<torch::Tensor> clss;
	std::vector<torch::Tensor> bbox_reg;
	std::vector<torch::Tensor> agn_hms;

	for (int i = 0; i < x.size(); i++) {
		//auto feature = x[i].to(torch::kCPU);
		auto feature = x[i];
		
		auto share_tower = m_tower[m_tower_names[2]];
		if (share_tower.get()->size() != 0) {
			feature = share_tower->forward(feature);
		}

		auto cls_tower = m_tower[m_tower_names[0]];
		torch::Tensor cls;
		if (cls_tower.get()->size() != 0) {
			cls = cls_tower->forward(feature);
		}

		auto bbox_tower = m_tower[m_tower_names[1]];
		auto bbox = bbox_tower->forward(feature);

		if (m_only_proposal == false) {
			clss.push_back(m_cls_logits->forward(cls));
		}

		if (m_with_agn_hm) {
			agn_hms.push_back(m_agn_hm->forward(bbox));
		}

		auto reg = m_bbox_pred(bbox);
		reg = m_scales[i]->as<Scale>()->forward(reg);
		auto relu = torch::nn::ReLU();
		bbox_reg.push_back(relu->forward(reg));
	}
	ret.clss = clss;
	ret.agn_hms = agn_hms;
	ret.bbox_reg = bbox_reg;

	return ret;
}