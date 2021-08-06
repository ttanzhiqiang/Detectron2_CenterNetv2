#include "Base.h"
#include "DLAStem.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DLAStemImpl::DLAStemImpl(int in_channels, int out_channels, int stride, BatchNorm::Type norm):
	DLACNNBaseImpl(in_channels, out_channels, stride),
	m_convbn1(nn::Conv2dOptions(in_channels, out_channels, 7).stride(1).padding(3).bias(false), norm) {
	register_module("conv1", m_convbn1);
}

void DLAStemImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	m_convbn1->initialize(importer, prefix + ".conv1", ModelImporter::kCaffe2MSRAFill);
}

torch::Tensor DLAStemImpl::forward(torch::Tensor x, torch::Tensor residual) {
	x = m_convbn1(x);
	x = relu_(x);
	return x;
}
