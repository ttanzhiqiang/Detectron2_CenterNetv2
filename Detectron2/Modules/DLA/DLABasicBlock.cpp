#include "Base.h"
#include "DLABasicBlock.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DLABasicBlockImpl::DLABasicBlockImpl(int in_channels, int out_channels, int stride, BatchNorm::Type norm) :
	DLACNNBaseImpl(in_channels, out_channels, stride),
	m_convbn1(nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1).bias(false), norm),
	m_convbn2(nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1).bias(false), norm) {
	register_module("conv1", m_convbn1);
	register_module("conv2", m_convbn2);
}

void DLABasicBlockImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	m_convbn1->initialize(importer, prefix + ".conv1", ModelImporter::kCaffe2MSRAFill);
	m_convbn2->initialize(importer, prefix + ".conv2", ModelImporter::kCaffe2MSRAFill);
}

torch::Tensor DLABasicBlockImpl::forward(torch::Tensor x, torch::Tensor residual) {
	if (residual.numel() == 0) {
		residual = x.clone();
	}
	auto out = m_convbn1(x);
	out = relu(out);
	out = m_convbn2(out);

	out += residual;
	out = relu(out);
	return out;
}

