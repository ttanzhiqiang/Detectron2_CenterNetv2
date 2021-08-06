#pragma 
#include "Base.h"
#include "DLACNNBase.h"
#include <Detectron2/Modules/Backbone.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/dla.py

	class SingleBiFPNImpl : public torch::nn::Module {
	public:
		/**
			in_channels (int): Number of input channels.
			out_channels (int): Number of output channels.
			stride (int): Stride for the first conv.
			norm (str or callable): normalization for all conv layers.
				See :func:`layers.get_norm` for supported format.
		*/
		SingleBiFPNImpl(std::vector<int> in_channels_list, int out_channels, BatchNorm::Type norm);

		std::vector<torch::nn::Sequential> output() {
			std::vector<torch::nn::Sequential> res(m_outputs.begin() + 1, m_outputs.end());
			return res;
		};

		void initialize(const ModelImporter& importer, const std::string& prefix);
		//virtual TensorMap forward(torch::Tensor x) override;
		TensorVec forward(TensorVec x);

	private:
		std::vector<std::vector<int>> m_nodes;
		std::vector<torch::nn::Sequential> m_outputs;
		std::map<std::string, torch::nn::Sequential> m_stages;
		std::map<std::string, torch::Tensor> m_stages_weight;
		int m_out_channels;
	};
	TORCH_MODULE(SingleBiFPN);
}