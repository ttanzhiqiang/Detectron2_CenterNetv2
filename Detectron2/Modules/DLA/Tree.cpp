#include "Base.h"
#include "Tree.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TreeImpl::TreeImpl(int levels, DLABasicBlock block, int in_channels, int out_channels, int stride, bool level_root,
	int root_dim, int root_kernel_size, int dilation , bool root_residual, BatchNorm::Type norm):
	m_level_root(level_root), m_levels(levels), m_out_channels(out_channels), m_stride(stride) {
	if (root_dim==0){
		root_dim = out_channels*2;
	}
	if (level_root & (levels > 1)) {
		root_dim += in_channels;
	}
	if (out_channels > 256) {
		root_dim += in_channels;
	}

	if (levels==1){
		m_DLABasicBlock1 = DLABasicBlock(in_channels, out_channels, stride, norm);
		m_DLABasicBlock2 = DLABasicBlock(out_channels, out_channels, 1, norm);
		register_module("tree1", m_DLABasicBlock1);
		register_module("tree2", m_DLABasicBlock2);
	}
	else{
		m_tree1 = shared_ptr<TreeImpl>(new TreeImpl(levels - 1, block, in_channels, out_channels, stride, false, 0,
				root_kernel_size, dilation, root_residual, norm));
		m_tree2 = shared_ptr<TreeImpl>(new TreeImpl(levels - 1, block, out_channels, out_channels, 1, false, root_dim + out_channels,
			root_kernel_size, dilation, root_residual, norm));
	
		register_module("tree1", m_tree1);
		register_module("tree2", m_tree2);
	}		

	
	if (levels==1){
		m_root = Root(root_dim, out_channels, root_kernel_size, root_residual, norm);
		register_module("root", m_root);
	}
	
	if (stride>1){
		m_downsample = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(stride));
		register_module("downsample", m_downsample);
	}
	
	if (in_channels != out_channels){
		m_project = ConvBn2d(nn::Conv2dOptions(in_channels, out_channels, 1).stride(1).bias(false), norm);
		register_module("project", m_project);
	}
}

void TreeImpl::initialize(const ModelImporter& importer, const std::string& prefix) {
	if (m_project) {
		m_project->initialize(importer, prefix + ".project", ModelImporter::kCaffe2MSRAFill);
	}
	if (m_root){
		m_root->initialize(importer, prefix + ".root");
	}
	if (m_levels == 1)
	{
		m_DLABasicBlock1->initialize(importer, prefix + ".tree1");
		m_DLABasicBlock1->initialize(importer, prefix + ".tree2");
	}
	else {
		m_tree1->initialize(importer, prefix + ".tree1");
		m_tree2->initialize(importer, prefix + ".tree2");
	}
}

void TreeImpl::freeze() {
	for (auto p : parameters()) {
		p.set_requires_grad(false);
	}
	auto self = shared_from_this();
	//FrozenBatchNorm2dImpl::convert_frozen_batchnorm(self);
}

torch::Tensor TreeImpl::forward(torch::Tensor x, torch::Tensor residual, std::vector<torch::Tensor> children) {
	std::vector<torch::Tensor> m_children;
	if (children.size() == 0) {
		m_children = {};
	}
	else {
		m_children = children;
	}

	torch::Tensor bottom;
	if (m_downsample.is_empty()) {
		bottom = x;
	}
	else {
		bottom = m_downsample->forward(x);
	}

	torch::Tensor m_residual;
	if (m_project.is_empty()) {
		m_residual = bottom;
	}
	else {
		m_residual = m_project->forward(bottom);
	}
		
	if (m_level_root){
		m_children.push_back(bottom);
	}

	torch::Tensor out,a;
	if (m_levels == 1){
		std::vector<torch::Tensor> root_in;
		auto x1 = m_DLABasicBlock1->forward(x, m_residual);
		auto x2 = m_DLABasicBlock2->forward(x1);
		root_in.push_back(x2);
		root_in.push_back(x1);
		for (int i = 0; i < m_children.size(); i++) {
			root_in.push_back(m_children[i]);
		}
				
		out = m_root->forward(root_in);
	}
	else{
		auto x1 = m_tree1->forward(x, m_residual);
		m_children.push_back(x1);
		out = m_tree2->forward(x1, a, m_children);
	}
	
	return out;
}
