#pragma 
#include "DLA.h"
#include "BiFPN.h"
#include "Base.h"
#include "DLACNNBase.h"
#include <Detectron2/Modules/Conv/ConvBn2d.h>
#include <Detectron2/Modules/BatchNorm/BatchNorm.h>
#include <Detectron2/Modules/Backbone.h>

namespace Detectron2
{
	struct CenterNetVec {
		std::vector<torch::Tensor> clss;
		std::vector<torch::Tensor> bbox_reg;
		std::vector<torch::Tensor> agn_hms;
	};

	struct convs_deform {
		int num_convs;
		bool use_deformable;
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/dla.py
	class Scale : public torch::nn::Module {
	public:
		Scale(float init_value=1.0) : m_init_value(init_value) {
			float* array = new float[1];
			array[0] = init_value;
			m_scale = torch::from_blob(array, { 1 }, dtype(torch::kFloat));
		};
		torch::Tensor forward(torch::Tensor x) {
			return x * (m_scale.to(x.device()));
		}
	private:
		torch::Tensor m_scale;
		float m_init_value;
	};

	class CenterNetHeadImpl : public torch::nn::Module {
	public:
		/**
			in_channels (int): Number of input channels.
			out_channels (int): Number of output channels.
			stride (int): Stride for the first conv.
			norm (str or callable): normalization for all conv layers.
				See :func:`layers.get_norm` for supported format.
		*/
		CenterNetHeadImpl() {};
		CenterNetHeadImpl(CfgNode &cfg, const ShapeSpec::Map& input_shape);
		virtual ~CenterNetHeadImpl() {}
		void initialize(const ModelImporter& importer, const std::string& prefix);
		CenterNetVec forward(TensorVec x);

	private:
		std::vector<std::string> m_tower_names;
		std::unordered_map < std::string, convs_deform > m_head_configs;
		std::unordered_map < std::string, torch::nn::Sequential> m_tower;
		//torch::nn::Sequential m_share_tower = { nullptr };
		//torch::nn::Sequential m_cls_tower = { nullptr };
		ConvBn2d m_bbox_tower = { nullptr };
		ConvBn2d m_conv = { nullptr };
		torch::nn::ModuleList m_scales;
		torch::nn::Conv2d m_bbox_pred = { nullptr };
		torch::nn::Conv2d m_agn_hm = { nullptr };
		torch::nn::Conv2d m_cls_logits = {nullptr};
		bool m_with_agn_hm;
		bool m_only_proposal;
		float m_bias_value;
	};
	TORCH_MODULE(CenterNetHead);
}