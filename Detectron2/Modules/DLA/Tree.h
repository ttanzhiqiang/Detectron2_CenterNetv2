#pragma once

#include "DLACNNBase.h"
#include "DLABasicBlock.h"
#include "Root.h"

#include <Detectron2/Modules/BatchNorm/FrozenBatchNorm2d.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/resnet.py

	/**
		The standard Tree used by DLA defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
		1x1, 3x3, 1x1, and a projection shortcut if needed.
	*/
	class TreeBaseImpl : public torch::nn::Module {
	public:
		TreeBaseImpl() {};
		TreeBaseImpl(int levels, DLABasicBlock block, int in_channels, int out_channels, int stride = 1, bool level_root = false,
			int root_dim = 0, int root_kernel_size = 1, int dilation = 1, bool root_residual = false, BatchNorm::Type norm = BatchNorm::kBN) {};
		virtual void initialize(const ModelImporter& importer, const std::string& prefix) = 0;
		virtual torch::Tensor forward(torch::Tensor x, torch::Tensor residual = {}, std::vector<torch::Tensor> children = {}) = 0;
	};
	TORCH_MODULE(TreeBase);

	class TreeImpl : public torch::nn::Module {
	public:
		TreeImpl(int levels, DLABasicBlock block, int in_channels, int out_channels, int stride = 1, bool level_root = false,
			int root_dim = 0, int root_kernel_size = 1, int dilation = 1, bool root_residual = false, BatchNorm::Type norm = BatchNorm::kBN);
		virtual ~TreeImpl() {}
		int stride() const { return m_stride; }
		int out_channels() const { return m_out_channels; }

		void initialize(const ModelImporter& importer, const std::string& prefix);
		torch::Tensor forward(torch::Tensor x, torch::Tensor residual = {}, std::vector<torch::Tensor> children = {});

		void freeze();

	private:
		DLABasicBlock m_DLABasicBlock1{ nullptr };
		DLABasicBlock m_DLABasicBlock2{ nullptr };

		/*std::shared_ptr<TreeBaseImpl> m_tree1;
		std::shared_ptr<TreeBaseImpl> m_tree2;*/
		std::shared_ptr<TreeImpl> m_tree1;
		std::shared_ptr<TreeImpl> m_tree2;

		Root m_root{ nullptr };
		ConvBn2d m_project{ nullptr };
		torch::nn::MaxPool2d m_downsample{ nullptr };
		bool m_level_root;
		int m_levels;
		int m_out_channels;
		int m_stride;
	};
	TORCH_MODULE(Tree);
}