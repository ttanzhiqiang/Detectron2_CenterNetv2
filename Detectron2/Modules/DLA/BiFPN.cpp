#include <assert.h>
#include <string>
#include "Base.h"
#include "BiFPN.h"
#include<math.h>
#include<memory>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BiFPNImpl::BiFPNImpl(Backbone bottom_up, std::vector<std::string> in_features, int out_channels, int num_repeats, BatchNorm::Type norm) :
	m_bottom_up(bottom_up), m_in_features(in_features), m_out_channels(out_channels), m_num_repeats(num_repeats){
	
	register_module("bottom_up", m_bottom_up);

	const ShapeSpec::Map& input_shapes = bottom_up->output_shapes();
	std::vector<int> in_channels, in_strides;
	in_channels.reserve(in_features.size());
	in_strides.reserve(in_features.size());
	for (auto& f : in_features) {
		auto iter = input_shapes.find(f);
		assert(iter != input_shapes.end());
		auto& item = iter->second;
		in_channels.push_back(item.channels);
		in_strides.push_back(item.stride);
	}

	//m_stages.reserve(in_strides.size());
	for (int i = 0; i < in_strides.size(); i++) {
		m_stages.push_back(IntLog2(in_strides[i]));
	}

	for (int i = 0; i < m_num_repeats; i++) {
		auto singleBiFPN = SingleBiFPN(in_channels, out_channels, norm);
		//std::cout << singleBiFPN << std::endl;
		m_repeated_bifpn.push_back(singleBiFPN);
		register_module(FormatString("repeated_bifpn%d",i), singleBiFPN);
		for (int j = 0; j < in_channels.size(); j++) {
			in_channels[j] = out_channels;
		}
	}
	SingleBiFPN out_features = m_repeated_bifpn[m_num_repeats - 1];
	//std::cout << out_features->output() << std::endl;
	
	int index = 0;
	for (int i = 0; i < in_features.size(); i++) {
		auto name = FormatString("dla%d", i + 3);
		register_module(name, out_features->output()[i]);
		auto& shape = m_output_shapes[name];
		shape.stride = in_strides[i];
		shape.channels = out_channels;
		shape.index = index++;
	}

	m_size_divisibility = in_strides[in_strides.size() - 1];
}

void BiFPNImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	for (int i = 0; i < m_num_repeats; i++) {
		std::string name = to_string(i);
		m_repeated_bifpn[i]->initialize(importer, prefix + name);
	}
}

TensorMap BiFPNImpl::forward(torch::Tensor x) {
	//std::cout << x.sizes() << std::endl;
	TensorMap bottom_up_features = m_bottom_up->forward(x);

	// Reverse feature maps into top-down order (from low to high resolution)
	TensorVec features;
	features.reserve(m_in_features.size());
	for (int i = 0; i < m_in_features.size(); i++) {
		auto& name = m_in_features[i];
		auto iter = bottom_up_features.find(name);
		assert(iter != bottom_up_features.end());
		features.push_back(iter->second);
	}

	TensorVec results;
	results.reserve(m_in_features.size());
	for (int i = 0; i < m_num_repeats; i++) {
		results = m_repeated_bifpn[i]->forward(features);
		features = results;
	}

	assert(results.size() == m_output_shapes.size());
	TensorMap ret;
	for (int i = 0; i < m_in_features.size(); i++) {
		ret[m_in_features[i]] = results[i];
	}
	return ret;
}