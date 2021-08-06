#include "Base.h"
#include "Level.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

LevelImpl::LevelImpl(int in_channels, int out_channels, int convs, BatchNorm::Type norm, int stride, int dilation) :
	DLACNNBaseImpl(in_channels, out_channels, stride), m_convs(convs) {
	for (int i = 0; i < convs; i++) {
		//auto m_convbn1 = ConvBn2d(in_channels, out_channels, 3 ,norm);
		auto conv_bn = ConvBn2d(nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1).bias(false), norm);
		m_layers.push_back(conv_bn);
		auto name = FormatString("conv%d", i+1);
		register_module(name, conv_bn);
		in_channels = out_channels;
	}
}

void LevelImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	for (int i = 0; i < m_convs; i++) {
		auto name = FormatString(".conv%d", (i+1));
		m_layers[i]->initialize(importer, prefix + name, ModelImporter::kCaffe2MSRAFill);
	}
}

torch::Tensor LevelImpl::forward(torch::Tensor x, torch::Tensor residual) {
	torch::Tensor out;
	for (int i = 0; i < m_convs; i++) {
		out = m_layers[i]->forward(x);
		x = out;
	}
	return out;
}
