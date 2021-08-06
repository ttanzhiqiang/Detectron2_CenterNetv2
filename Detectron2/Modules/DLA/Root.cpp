#include "Base.h"
#include "Root.h"
#include <Detectron2/Modules/BatchNorm/FrozenBatchNorm2d.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RootImpl::RootImpl(int in_channels, int out_channels, int stride, bool residual, BatchNorm::Type norm, int kernel_size) :
	m_in_channels(in_channels), m_out_channels(out_channels), m_stride(stride), m_residual(residual),
	m_convbn1(nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(stride).padding((kernel_size-1)/2).bias(false), norm){
	register_module("conv1", m_convbn1);
}

void RootImpl::initialize(const ModelImporter& importer, const std::string& prefix) {
	m_convbn1->initialize(importer, prefix + ".conv1", ModelImporter::kCaffe2MSRAFill);
}

void RootImpl::freeze() {
	for (auto p : parameters()) {
		p.set_requires_grad(false);
	}
	auto self = shared_from_this();
	//FrozenBatchNorm2dImpl::convert_frozen_batchnorm(self);
}

torch::Tensor RootImpl::forward(std::vector<torch::Tensor>& x) {
	auto children = x;
	auto m_x = torch::cat(x, 1);
	//std::cout << m_x.sizes() << std::endl;
	auto out = m_convbn1(m_x);

	if (m_residual) {
		out += children[0];
	}
	out = relu(out);
	
	return out;
}
