#pragma once

#include <Detectron2/Structures/ShapeSpec.h>
#include <Detectron2/Modules/Conv/ConvBn2d.h>


namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/resnet.py

	/**
		The basic root block for DLA with two 3x3 conv layers and a projection shortcut if needed.
	*/
	class RootImpl : public torch::nn::Module {
	public:
		/**
			in_channels (int): Number of input channels.
			out_channels (int): Number of output channels.
			stride (int): Stride for the first conv.
			norm (str or callable): normalization for all conv layers.
				See :func:`layers.get_norm` for supported format.
		*/
		RootImpl(int in_channels, int out_channels, int stride, bool residual, BatchNorm::Type norm, int kernel_size = 1);
		virtual ~RootImpl() {}

		int stride() const { return m_stride; }
		int out_channels() const { return m_out_channels; }

		void initialize(const ModelImporter &importer, const std::string &prefix);
		torch::Tensor forward(std::vector<torch::Tensor>& x);

		void freeze();

	private:
		bool m_residual;
		int m_in_channels;
		int m_out_channels;
		int m_stride;
		ConvBn2d m_convbn1;
	};
	TORCH_MODULE(Root);
}