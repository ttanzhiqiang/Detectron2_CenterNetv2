#include "Base.h"
#include <iostream>
#include <fstream>
#include "Utils.h"
//#include "Structures/Instances.h"
#include "Visualizer.h"
#include "DefaultTrainer.h"
#include <Detectron2/Utils/Timer.h>
#include <Detectron2/Data/ResizeShortestEdge.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

DefaultTrainer::DefaultTrainer(const CfgNode& cfg) : m_model(nullptr) 
{
	m_cfg = cfg.clone();  // cfg can be modified by model
	{
		Timer timer("build_model");
		m_model = build_model(m_cfg);
	}

	auto train_json = CfgNode::parseTuple<string>(cfg["DATASETS.TRAIN"], { "" })[0];
	auto train_path = CfgNode::parseTuple<string>(cfg["DATASETS.TRAIN"], { "" })[1];
	batch_size = cfg["SOLVER.IMS_PER_BATCH"].as<int>();
	base_lr = cfg["SOLVER.BASE_LR"].as<float>();
	base_momentum = cfg["SOLVER.MOMENTUM"].as<float>();
	base_weight_decay = cfg["SOLVER.WEIGHT_DECAY"].as<float>();
	max_iter = cfg["SOLVER.MAX_ITER"].as<int>();
	_warmup_start = cfg["SOLVER.WARMUP_FACTOR"].as<float>();
	_warmup_steps = cfg["SOLVER.WARMUP_ITERS"].as<float>();
	_device = torch::Device(DeviceType::CUDA, 0);
	auto steps = CfgNode::parseTuple<string>(cfg["SOLVER.STEPS"], { "" });
	for (int i = 0; i < steps.size(); i++)
	{
		decay_step.push_back(std::stoi(steps[i]));
	}
	auto m_max_size_train = cfg["INPUT.MAX_SIZE_TRAIN"].as<int>();
	auto m_random_flip = cfg["INPUT.RANDOM_FLIP"].as<string>();

	_dataset = std::make_shared<CocoDataset>(train_path, train_json, m_max_size_train, m_random_flip);
	auto train_coco_ann = _dataset->coco_ann();
	train_path_size = train_coco_ann.iname2iid.size();
	total_epochs = (max_iter * batch_size / train_path_size) + 1;
	std::vector<int> _decay_epochs;
	for (int i = 0; i < decay_step.size(); i++)
	{
		_decay_epochs.push_back(decay_step[i] * batch_size / train_path_size);
	}
	decay_epochs = std::set<int>(_decay_epochs.begin(), _decay_epochs.end());


	auto optim_opts = torch::optim::SGDOptions(base_lr).momentum(base_momentum).weight_decay(base_weight_decay);
	// construct SGD optimizer
	_optimizer = std::make_shared<torch::optim::SGD>(m_model->parameters(), optim_opts);

	m_metadata = MetadataCatalog::get("coco_2017_train");
	{
		Timer timer("load_checkpoint");
		//m_model->load_checkpoint(cfg["MODEL.WEIGHTS"].as<string>(""), false);
	}

	m_transform_gen = shared_ptr<TransformGen>(new ResizeShortestEdge(
		{ cfg["INPUT.MAX_SIZE_TRAIN"].as<int>(), cfg["INPUT.MAX_SIZE_TRAIN"].as<int>()},
		cfg["INPUT.MAX_SIZE_TRAIN"].as<int>()
	));

	m_input_format = cfg["INPUT.FORMAT"].as<string>();
	assert(m_input_format == "RGB" || m_input_format == "BGR");
}

void DefaultTrainer::LoadData(std::vector<Detectron2::ImgData>& img_datas, std::vector<DatasetMapperOutput>& inputs, int& img_data_i)
{
	for (auto img_data : img_datas)
	{
		if (img_data_i < img_datas.size())
		{
			img_data.to(_device);
			inputs[img_data_i].image = img_data.img_tsr/*.requires_grad_()*/;
			inputs[img_data_i].height = make_shared<int>(img_data.pad_shape[0]);
			inputs[img_data_i].width = make_shared<int>(img_data.pad_shape[1]);
			ImageSize m_ImageSize;
			m_ImageSize.height = img_data.pad_shape[0];
			m_ImageSize.width = img_data.pad_shape[1];
			Instances m_instances(m_ImageSize);
			m_instances.set("gt_classes", img_data.gt_labels);
			m_instances.set("gt_boxes", img_data.gt_bboxes/*.requires_grad_()*/);
			//cout << img_data.gt_bboxes << endl;
			/*
			int max_num = 0;
			for (int i = 0; i < img_data.gt_seg.size(); i++)
			{
				if (max_num < img_data.gt_seg[i].numel())
				{
					max_num = img_data.gt_seg[i].numel();
				}
			}
			std::vector<torch::Tensor> m_gt_seg;
			for (int i = 0; i < img_data.gt_seg.size(); i++)
			{
				cout << "img_data.gt_seg[i]_size:" << img_data.gt_seg[i].sizes() << endl;
				cout << "img_data.gt_seg[i]:" << img_data.gt_seg[i] << endl;
				//torch::Tensor b = torch::ones(max_num);
				auto m_gt_seg_i = img_data.gt_seg[i].expand({1,max_num });
				m_gt_seg.push_back(m_gt_seg_i);
			}
			*/
			//torch::Tensor gt_seg_cat = torch::stack({ img_data.gt_seg }, 0);
			//InstancesPtr gt_results(new Instances(m_ImageSize));
			//gt_results->set("gt_masks", gt_seg_cat);
			//m_instances.set("gt_masks", gt_seg_cat);
			inputs[img_data_i].instances = make_shared<Instances>(m_instances);
			img_data_i++;

			//cout << "img_data.gt_bboxes:" << img_data.gt_bboxes << endl;
			//int box_num = img_data.gt_bboxes.size(0);
			//for (int i_box_num = 0; i_box_num < box_num; i_box_num++)
			//{
			//	auto b = tolist(img_data.gt_bboxes[i_box_num]);
			//	cv::Rect rect;
			//	auto x = (int)b[0].item<float>();
			//	auto y = (int)b[1].item<float>();
			//	auto w = (int)b[2].item<float>() - x;
			//	auto h = (int)b[3].item<float>() - y;
			//	rect = cv::Rect(x, y, w, h);
			//	cv::rectangle(img_data.img_cv2, rect, cv::Scalar(255, 0, 0), 2);
			//}
			//cv::imwrite("F:\\data\\faster_rcnn\\output\\centernetv2\\" + img_data.file_name, img_data.img_cv2);
		}
	}
}


void DefaultTrainer::LoadData_random(std::vector<Detectron2::ImgData>& img_datas, std::vector<DatasetMapperOutput>& inputs, int& img_data_i)
{
	for (int i = 0; i < batch_size; i++)
	{
		int random_i = std::rand() % img_datas.size();
		auto img_data = img_datas[random_i];
		img_data.to(_device);
		inputs[img_data_i].image = img_data.img_tsr/*.requires_grad_()*/;
		inputs[img_data_i].height = make_shared<int>(img_data.pad_shape[0]);
		inputs[img_data_i].width = make_shared<int>(img_data.pad_shape[1]);
		ImageSize m_ImageSize;
		m_ImageSize.height = img_data.pad_shape[0];
		m_ImageSize.width = img_data.pad_shape[1];
		Instances m_instances(m_ImageSize);
		m_instances.set("gt_classes", img_data.gt_labels);
		m_instances.set("gt_boxes", img_data.gt_bboxes/*.requires_grad_()*/);
		inputs[img_data_i].instances = make_shared<Instances>(m_instances);
		img_data_i++;

		int box_num = img_data.gt_bboxes.size(0);
		for (int i_box_num = 0; i_box_num < box_num; i_box_num++)
		{
			auto b = tolist(img_data.gt_bboxes[i_box_num]);
			cv::Rect rect;
			auto x = (int)b[0].item<float>();
			auto y = (int)b[1].item<float>();
			auto w = (int)b[2].item<float>() - x;
			auto h = (int)b[3].item<float>() - y;
			rect = cv::Rect(x, y, w, h);
			cv::rectangle(img_data.img_cv2, rect, cv::Scalar(255, 0, 0), 2);
		}
		cv::imwrite("F:\\data\\faster_rcnn\\output\\centernetv2\\" + img_data.file_name, img_data.img_cv2);
	}
}

void DefaultTrainer::train() {

   //print model parameters
	auto model_dict = m_model->named_parameters(true);
	for (auto n: model_dict)
	{
		cout << setw(50) << n.key() << ";" << setw(20) << n.value().sizes() << "\n";
	}
	auto model_dict_buffer = m_model->named_buffers(true);
	for (auto n : model_dict_buffer)
	{
		cout << setw(50) << n.key() << ";" << setw(20) << n.value().sizes() << "\n";
	}
	m_model->to(_device);
	_pg_tracker = ProgressTracker(total_epochs, _dataset->size().value());
	m_model->train();
	std::cout << m_model << std::endl;
	auto loader_opts = torch::data::DataLoaderOptions().batch_size(batch_size).workers(4);
	auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
		(std::move(*_dataset), loader_opts);
	float loss_sum = 0;
	//InstancesPtr predictions;
	
	for (int64_t epoch = 0; epoch <= total_epochs; epoch++) {
		// check if lr needs to be decayed
		if (decay_epochs.find(epoch) != decay_epochs.end()) {
			base_lr *= 0.1;
			set_lr(base_lr);
		}
		// iterate over all image data
		for (auto& img_datas : *dataloader) {
			// check if lr needs to be warmed up at the begining
			std::vector<DatasetMapperOutput> inputs(batch_size);
			warmup_lr();
			if (img_datas.size() ==  batch_size)
			{
				int img_data_i = 0;
				LoadData(img_datas, inputs, img_data_i);
			}
			else
			{
				//to do
				int img_data_i = 0;
				LoadData_random(img_datas, inputs, img_data_i);
			}
			TensorMap m_losses;
			{
				auto InstancesMap = m_model->forward(inputs);
				//predictions = get<0>(InstancesMap)[0];
				m_losses = get<1>(InstancesMap);
			}
			auto tot_loss = sum_loss(m_losses);
			_pg_tracker.track_loss(m_losses);
			_optimizer->zero_grad();
			tot_loss.backward();
			_optimizer->step();
			_pg_tracker.next_iter();
			loss_sum = tot_loss.item().toFloat();
			float m_lr = get_lr();
			std::cout << "Epoch: " << epoch << "," << "iter" << _pg_tracker.cur_iter() << ","
				<< " Training Loss: " << loss_sum << "," 
				<< "m_lr:" << double(m_lr) << endl;
			inputs.clear();
		}
		_pg_tracker.next_epoch();
		if (epoch % 100 == 0 && epoch != 0) {
			torch::serialize::OutputArchive archive;
			m_model->save(archive);
			archive.save_to("F:\\data\\faster_rcnn\\weight\\epoch_" + std::to_string(epoch) + ".pt");
			//torch::save(m_model, "F:\\data\\faster_rcnn\\weight\\resnet101_epoch_" +  std::to_string(epoch) + ".pt");
		}
	}
	return;
}



void DefaultTrainer::warmup_lr() {
	auto iters = _pg_tracker.cur_iter();
	if (iters <= _warmup_steps) {
		float lr = _warmup_start * base_lr + (1 - _warmup_start) * iters / _warmup_steps * base_lr;
		set_lr(lr);
	}
}


void DefaultTrainer::set_lr(float lr) {
	for (auto& group : _optimizer->param_groups()) {
		static_cast<torch::optim::SGDOptions&>(group.options()).lr(lr);
	}
}

float DefaultTrainer::get_lr() {
	return static_cast<torch::optim::SGDOptions&>(_optimizer->param_groups()[0].options()).lr();
}

torch::Tensor DefaultTrainer::sum_loss(TensorMap& loss_map) {
	auto tot_loss = torch::tensor(0, torch::TensorOptions().dtype(torch::kFloat32)).to(_device);
	for (auto& loss : loss_map) {
		auto name = loss.first;
		tot_loss += loss.second;
		cout << name << ":" << loss.second.item().toFloat() << endl;
	}
	return tot_loss;
}