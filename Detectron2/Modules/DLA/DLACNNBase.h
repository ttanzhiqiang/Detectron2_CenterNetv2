#pragma once

#include <Detectron2/Structures/ShapeSpec.h>
#include <Detectron2/Modules/Conv/ConvBn2d.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from layers/blocks.py

	/**
		A CNN block is assumed to have input channels, output channels and a stride.
		The input and output of `forward()` method must be NCHW tensors.
		The method can perform arbitrary computation but must match the given
		channels and stride specification.
	*/
	class DLACNNBaseImpl : public torch::nn::Module {
	public:
		/**
			The `__init__` method of any subclass should also contain these arguments.

			Args:
				in_channels (int):
				out_channels (int):
				stride (int):
		*/
		DLACNNBaseImpl(int in_channels, int out_channels, int stride, int kernel_size = 1);
		virtual ~DLACNNBaseImpl() {}

		int stride() const { return m_stride; }
		int out_channels() const { return m_out_channels; }

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) = 0;
		//virtual torch::Tensor forward(torch::Tensor x) = 0;
		virtual torch::Tensor forward(torch::Tensor x, torch::Tensor residual) = 0;
		/**
			Make this block not trainable.
			This method sets all parameters to `requires_grad=False`,
			and convert all BatchNorm layers to FrozenBatchNorm
		*/
		void freeze();

	protected:
		int m_in_channels;
		int m_out_channels;
		int m_stride;
	};
	TORCH_MODULE(DLACNNBase);
}