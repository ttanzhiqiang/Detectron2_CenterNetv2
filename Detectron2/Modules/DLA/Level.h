#pragma once

#include "DLACNNBase.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/resnet.py

	/**
		The basic residual block for ResNet-18 and ResNet-34 defined in :paper:`ResNet`,
		with two 3x3 conv layers and a projection shortcut if needed.
	*/
	class LevelImpl : public DLACNNBaseImpl {
	public:
		/**
			in_channels (int): Number of input channels.
			out_channels (int): Number of output channels.
			stride (int): Stride for the first conv.
			norm (str or callable): normalization for all conv layers.
				See :func:`layers.get_norm` for supported format.
		*/
		LevelImpl(int in_channels, int out_channels, int convs, BatchNorm::Type norm, int stride = 1, int dilation = 1);
		
		int convs() const { return m_convs; }

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;
		virtual torch::Tensor forward(torch::Tensor x, torch::Tensor residual = {}) override;

	private:
		int m_convs;
		//ConvBn2d m_convbn1;
		std::vector<ConvBn2d> m_layers;
	};
	TORCH_MODULE(Level);
}