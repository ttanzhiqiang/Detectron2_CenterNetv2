#include "Base.h"
#include "DLA.h"

#include "DLABasicBlock.h"
#include "DLACNNBase.h"
#include "DLAStem.h"
#include "Root.h"
#include "Tree.h"
#include "Level.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Backbone Detectron2::build_dla_backbone(CfgNode &cfg, const ShapeSpec &input_shape) {
	// need registration of new blocks/stems?
	auto a = cfg["MODEL.DLA.NORM"].as<string>();
	auto norm = BatchNorm::GetType(cfg["MODEL.DLA.NORM"].as<string>());
	auto freeze_at = cfg["MODEL.BACKBONE.FREEZE_AT"].as<int>();
	auto out_features_ = cfg["MODEL.CENTERNET.OUT_FEATURES"].as<vector<string>>();
	unordered_set<string> out_features;
	out_features.insert(out_features_.begin(), out_features_.end());

	auto depth = cfg["MODEL.CENTERNET.DEPTH"].as<int>();
	auto stem = DLAStem(input_shape.channels, cfg["MODEL.CENTERNET.STEM_OUT_CHANNELS"].as<int>(), 1, norm);
	auto in_channels = cfg["MODEL.CENTERNET.STEM_OUT_CHANNELS"].as<int>();
	auto level0 = Level(in_channels, in_channels, 1, norm,1);
	auto level1 = Level(in_channels, in_channels*2, 1, norm, 2);
	in_channels = in_channels * 2;
	auto out_channels = in_channels * 2;

	////std::vector<std::string>> m_out_features;
	//for(int i=0;i<6;i++){
	//	m_out_features.push_back(FormatString(".dla%d", i)
	//}

	auto num_blocks_per_stage = map<int, torch::IntList>{
		{ 34,  {1, 1, 1, 2, 2, 1}}
	}[depth];
	
	int max_stage_idx = 0;
	for (auto f : out_features) {
		int idx = unordered_map<string, int>{{"dla2", 2}, {"dla3", 3}, {"dla4", 4}, {"dla5", 5} }[f];
		if (idx > max_stage_idx) max_stage_idx = idx;
	}

	std::vector<std::vector<Tree>> stages(max_stage_idx - 1);
	bool level_root = false;
	for	(int stage_idx = 2; stage_idx< max_stage_idx+1; stage_idx++){
		if (stage_idx > 2) {
			level_root = true;
		}
		auto basicblock = DLABasicBlock(in_channels, out_channels, 1, norm);
		auto block = Tree(num_blocks_per_stage[stage_idx], basicblock, in_channels, out_channels, 2, level_root);
			
		stages[stage_idx-2].push_back(block);
		in_channels = out_channels;
		out_channels = in_channels * 2;
	}
	auto ret = make_shared<DLAImpl>(stem, level0, level1, stages, out_features);
	ret->freeze(freeze_at);
	return shared_ptr<BackboneImpl>(ret);
	//return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DLAImpl::DLAImpl(DLAStem stem, Level level0, Level level1, const std::vector<std::vector<Tree>>& stages,
	const std::unordered_set<std::string>& out_features, bool residual_root) : m_stem(stem), m_level0(level0),
	m_level1(level1), m_out_features(out_features) {
	register_module("base_layer", m_stem);
	register_module("level0", m_level0);
	register_module("level1", m_level1);

	m_output_shapes["base_layer"].stride = m_stem->stride();
	m_output_shapes["base_layer"].channels = m_stem->out_channels();
	m_output_shapes["level0"].stride = m_level0->stride();
	m_output_shapes["level0"].channels = m_level0->out_channels();
	m_output_shapes["level1"].stride = m_level1->stride();
	m_output_shapes["level1"].channels = m_level1->out_channels();

	int current_stride = m_level1->stride();
	m_names.reserve(stages.size());
	m_stages.reserve(stages.size());
	string name;
	int curr_channels;
	for (int i = 0; i < stages.size(); i++) {
		auto stage = torch::nn::Sequential();

		auto &blocks = stages[i];
		assert(!blocks.empty());
		for (int j = 0; j < blocks.size(); j++) {
			auto &block = blocks[j];
			current_stride *= block->stride();
			curr_channels = block->out_channels();
			stage->push_back(block);
		}

		name = FormatString("dla%d", i+2);
		m_stages.push_back(stage);
		register_module(name, stage);
		m_names.push_back(name);

		auto &spec = m_output_shapes[name];
		spec.stride = current_stride;
		spec.channels = curr_channels;

	}

	if (m_out_features.empty()) {
		m_out_features = { name };
	}
	unordered_set<string> children;
	for (auto iter : named_children()) {
		children.insert(iter.key());
	}
	for (auto out_feature : m_out_features) {
		assert(children.find(out_feature) != children.end());
	}
}


void DLAImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	m_stem->initialize(importer, prefix + ".base_layer");
	m_level0->initialize(importer, prefix + ".level0");
	m_level1->initialize(importer, prefix + ".level1");

	for (int stageIndex = 0; stageIndex < m_stages.size(); stageIndex++) {
		auto &stage = m_stages[stageIndex];
		for (auto &block : stage->children()) {
			block->as<Tree>()->initialize(importer,prefix + FormatString(".level%d", stageIndex));
		}
	}
}

TensorMap DLAImpl::forward(torch::Tensor x) {
	TensorMap outputs;
	x = m_stem->forward(x);
	outputs["base_layer"] = x;
	x = m_level0->forward(x);
	outputs["level0"] = x;
	x = m_level1->forward(x);
	outputs["level1"] = x;

	for (int i = 0; i < m_names.size(); i++) {
		auto &stage = m_stages[i];
		auto &name = m_names[i];
		torch::Tensor residual = {};
		std::vector<torch::Tensor> children = {};
		x = stage->forward(x, residual, children);
		if (m_out_features.find(name) != m_out_features.end()) {
			outputs[name] = x;
		}
	}
	return outputs;
}

std::shared_ptr<DLAImpl> DLAImpl::freeze(int freeze_at) {
	if (freeze_at >= 1) {
		m_stem->freeze();
	}
	for (int i = 0; i < m_stages.size(); i++) {
		auto &stage = m_stages[i];
		if (freeze_at >= i + 2) {
			for (auto &block : stage->children()) {
				block->as<Tree>()->freeze();
			}
		}
	}
	return dynamic_pointer_cast<DLAImpl>(shared_from_this());
}
