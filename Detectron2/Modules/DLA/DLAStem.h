#pragma once

#include "DLACNNBase.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/resnet.py

	// The standard ResNet stem (layers before the first residual block).
	class DLAStemImpl : public DLACNNBaseImpl {
	public:
		// norm (str or callable): norm after the first conv layer.
		//   See : func:`layers.get_norm` for supported format.
		DLAStemImpl(int in_channels, int out_channels, int stride, BatchNorm::Type norm);

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;
		virtual torch::Tensor forward(torch::Tensor x, torch::Tensor residual = {}) override;

	private:
		ConvBn2d m_convbn1;
	};
	TORCH_MODULE(DLAStem);
}