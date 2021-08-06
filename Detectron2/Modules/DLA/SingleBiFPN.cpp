#include <assert.h>
#include <string>
#include "Base.h"
#include "SingleBiFPN.h"

using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

SingleBiFPNImpl::SingleBiFPNImpl(std::vector<int> in_channels_list, int out_channels, BatchNorm::Type norm) : m_out_channels(out_channels) {
	//std::vector<int> in_channels_list = { 128, 256, 512 };
	//int out_channels;
	if (in_channels_list.size() == 5) {
		m_nodes = {
			{ 3, 3, 4},
			{ 2, 2, 5} ,
			{ 1, 1, 6} ,
			{ 0, 0, 7},
			{ 1, 1, 7, 8},
			{ 2, 2, 6, 9},
			{ 3, 3, 5, 10},
			{ 4, 4, 11}
		};
	}
	else if (in_channels_list.size() == 3) {
		m_nodes = {
			{ 1, 1, 2},
			{ 0, 0, 3},
			{ 1, 1, 3, 4},
			{ 2, 2, 5},
		};
	}

	std::vector<int> node_info;
	std::vector<int> num_output_connections;
	std::string name;
	std::vector<std::string> NameVec;
	for (int i = 0; i < in_channels_list.size(); i++) {
		node_info.push_back(in_channels_list[i]);
		num_output_connections.push_back(0);
	}
	for (int i = 0; i < m_nodes.size(); i++) {
		auto feat_level = m_nodes[i][0];
		std::vector<int> inputs_offsets;
		for (int j = 1; j < m_nodes[i].size(); j++) {
			inputs_offsets.push_back(m_nodes[i][j]);
		}

		string inputs_offsets_str;
		for (int j = 0; j < inputs_offsets.size() - 1; j++) {
			inputs_offsets_str += to_string(inputs_offsets[j]) + "_";
		}
		inputs_offsets_str += to_string(inputs_offsets[inputs_offsets.size() - 1]);

		for (int j = 0; j < inputs_offsets.size(); j++) {
			auto input_offset = inputs_offsets[j];
			num_output_connections[inputs_offsets[j]] += 1;

			auto in_channels = node_info[input_offset];
			if (in_channels != out_channels) {
				auto lateral_conv = ConvBn2d(nn::Conv2dOptions(in_channels, out_channels, 1),norm);
				name = "lateral_" + to_string(input_offset) + "_f" + to_string(feat_level);
				NameVec.push_back(name);
				m_stages.insert(make_pair(name, lateral_conv));
				auto nCount = std::count(NameVec.begin(), NameVec.end(), name);
				if (nCount==1) {         // if name does not exists,then registers.
					register_module(name, lateral_conv);
				}
			}
		}
		node_info.push_back(out_channels);
		num_output_connections.push_back(0);

		name = "weights_f" + to_string(feat_level) + "_" + inputs_offsets_str;
		auto weights = torch::ones(inputs_offsets.size());
		m_stages_weight.insert(make_pair(name, weights));

		name = "outputs_f" + to_string(feat_level) + "_" + inputs_offsets_str;
		auto output_conv = torch::nn::Sequential(ConvBn2d(nn::Conv2dOptions(out_channels, out_channels, 1), norm));
		m_outputs.push_back(output_conv);
		m_stages.insert(make_pair(name, output_conv));
		register_module(name, output_conv);
	}
}

void SingleBiFPNImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	//m_convbn1->initialize(importer, prefix + ".conv1", ModelImporter::kCaffe2MSRAFill);
	//m_convbn2->initialize(importer, prefix + ".conv2", ModelImporter::kCaffe2MSRAFill);
}


TensorVec SingleBiFPNImpl::forward(TensorVec x) {
	//std::reverse(x.begin(), x.end());
	std::vector<torch::Tensor> feats = x;

	auto num_levels = feats.size();
	std::vector<int> num_output_connections;
	num_output_connections.resize(feats.size());
	std::string name;
	for (int i = 0; i < m_nodes.size(); i++) {
		auto feat_level = m_nodes[i][0];
		std::vector<int> inputs_offsets;
		for (int j = 1; j < m_nodes[i].size(); j++) {
			inputs_offsets.push_back(m_nodes[i][j]);
		}

		string inputs_offsets_str;
		for (int j = 0; j < inputs_offsets.size() - 1; j++) {
			inputs_offsets_str += to_string(inputs_offsets[j]) + "_";
		}
		inputs_offsets_str += to_string(inputs_offsets[inputs_offsets.size() - 1]);

		std::vector<torch::Tensor> input_nodes;
		auto target_h = feats[feat_level].sizes()[2];
		auto target_w = feats[feat_level].sizes()[3];
		for (int j = 0; j < inputs_offsets.size(); j++) {
			auto input_offset = inputs_offsets[j];
			num_output_connections[inputs_offsets[j]] += 1;
			auto input_node = feats[inputs_offsets[j]];

			if (input_node.sizes()[1] != m_out_channels) {
				name = "lateral_" + to_string(input_offset) + "_f" + to_string(feat_level);
				input_node = m_stages[name]->forward(input_node);
			}

			auto h = input_node.sizes()[2];
			auto w = input_node.sizes()[3];
			if (h > target_h && w > target_w) {
				auto height_stride_size = ((h - 1) / target_h + 1);
				auto width_stride_size = ((w - 1) / target_w + 1);
				assert(height_stride_size == width_stride_size);
				auto pool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ height_stride_size + 1, width_stride_size + 1 })
					.stride({ height_stride_size ,width_stride_size }).padding(1));
				input_node = pool->forward(input_node);
			}
			else if (h <= target_h && w <= target_w) {
				if (h < target_h || w < target_w) {
					input_node = torch::nn::functional::interpolate(input_node, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{target_h, target_h} ).mode(torch::kNearest));
				}
			}
			input_nodes.push_back(input_node);
		}

		name = "weights_f" + to_string(feat_level) + "_" + inputs_offsets_str;
		auto weights = m_stages_weight[name];
		auto relu = torch::nn::ReLU();
		weights = relu->forward(weights);
		auto norm_weights = weights / (weights.sum() + 0.001);

		auto new_node = torch::stack(input_nodes, -1);
		auto device_ = new_node.device();
		new_node = (norm_weights.to(device_) * new_node).sum(-1);
		auto sig = torch::nn::Sigmoid();
		new_node = new_node * new_node * sig->forward(new_node);

		name = "outputs_f" + to_string(feat_level) + "_" + inputs_offsets_str;
		feats.push_back(m_stages[name]->forward(new_node));

		//std::cout << "new_node: " << new_node.sizes() << std::endl;

		num_output_connections.push_back(0);
	}

	TensorVec ret;
	auto reverse_m_nodes = m_nodes;
	std::reverse(reverse_m_nodes.begin(), reverse_m_nodes.end());
	for (int idx = 0; idx < num_levels; idx++) {
		for (int i = 0; i < reverse_m_nodes.size(); i++) {
			if (reverse_m_nodes[i][0] == idx) {
				//std::cout << m_nodes[idx][0] << "  " << -1 - i << std::endl;
				ret.push_back(feats[-1 - i + feats.size()]);
				break;
			}
		}		
	}
	return ret;
}
