#pragma 
#include "DLA.h"
#include "SingleBiFPN.h"
#include "Base.h"
#include "DLACNNBase.h"
#include <Detectron2/Modules/Backbone.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/dla.py

	class BiFPNImpl : public BackboneImpl {
	public:
		/**
			in_channels (int): Number of input channels.
			out_channels (int): Number of output channels.
			stride (int): Stride for the first conv.
			norm (str or callable): normalization for all conv layers.
				See :func:`layers.get_norm` for supported format.
		*/
		BiFPNImpl(Backbone bottom_up, std::vector<std::string> in_features, int out_channels, int num_repeats, BatchNorm::Type norm);

		virtual void initialize(const ModelImporter& importer, const std::string& prefix) override;
		virtual TensorMap forward(torch::Tensor x) override;

		virtual int size_divisibility() override { return m_size_divisibility; }

	private:
		Backbone m_bottom_up;
		std::vector<int> m_stages;
		std::vector<std::string> m_in_features;
		std::vector<SingleBiFPN> m_repeated_bifpn;
		int m_out_channels;
		int m_num_repeats;
		int m_size_divisibility;
	};
	TORCH_MODULE(BiFPN);
}