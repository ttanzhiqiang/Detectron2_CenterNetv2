#pragma once

#include <Detectron2/Modules/Backbone.h>
#include "DLACNNBase.h"
#include "DLAStem.h"
#include "Level.h"
#include "Tree.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/resnet.py

	// Implement :paper:`ResNet`.
	class DLAImpl : public BackboneImpl {
	public:
		/**
			stem (nn.Module): a stem module
			stages (list[list[CNNBlockBase]]): several (typically 4) stages,
				each contains multiple :class:`CNNBlockBase`.
			num_classes (None or int): if None, will not perform classification.
				Otherwise, will create a linear layer.
			out_features (list[str]): name of the layers whose outputs should
				be returned in forward. Can be anything in "stem", "linear", or "res2" ...
				If None, will return the output of the last layer.
		*/
		DLAImpl(DLAStem stem, Level level0, Level level1, const std::vector<std::vector<Tree>>& stages,
			const std::unordered_set<std::string>& out_features, bool residual_root = false);

		virtual void initialize(const ModelImporter& importer, const std::string& prefix) override;

		virtual TensorMap forward(torch::Tensor x) override;

		std::vector<torch::nn::Sequential> stages() { return m_stages; }

		std::shared_ptr<DLAImpl> freeze(int freeze_at = 0);

	private:
		DLAStem m_stem;
		Level m_level0;
		Level m_level1;
		std::unordered_set<std::string> m_out_features;

		std::vector<std::string> m_names;
		std::vector<torch::nn::Sequential> m_stages;
	};
	TORCH_MODULE(DLA);

	/**
		Create a ResNet instance from config.

		Returns:
			ResNet: a :class:`DLA` instance.
	*/
	Backbone build_dla_backbone(CfgNode& cfg, const ShapeSpec& input_shape);
}