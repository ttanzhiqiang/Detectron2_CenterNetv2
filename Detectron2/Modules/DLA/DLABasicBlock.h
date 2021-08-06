#pragma once

#include "DLACNNBase.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/dla.py

	/**
		The basic residual block for DLA.
	*/
	class DLABasicBlockImpl : public DLACNNBaseImpl {
	public:
		/**
			in_channels (int): Number of input channels.
			out_channels (int): Number of output channels.
			stride (int): Stride for the first conv.
			norm (str or callable): normalization for all conv layers.
				See :func:`layers.get_norm` for supported format.
		*/
		DLABasicBlockImpl(int in_channels, int out_channels, int stride, BatchNorm::Type norm);

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;
		virtual torch::Tensor forward(torch::Tensor x, torch::Tensor residual = {}) override;

	private:
		ConvBn2d m_convbn1;
		ConvBn2d m_convbn2;
	};
	TORCH_MODULE(DLABasicBlock);
}