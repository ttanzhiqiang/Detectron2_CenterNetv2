#include "Base.h"
#include "VisualizationDemo.h"
#include "coco/data.hpp"
#include <Detectron2/Data/BuiltinDataset.h>
#include <Detectron2/Data/MetadataCatalog.h>
#include <Detectron2/Utils/AsyncPredictor.h>
#include <Detectron2/Utils/DefaultPredictor.h>
#include <Detectron2/Utils/Utils.h>
#include <Detectron2/Utils/VideoVisualizer.h>
#include <Detectron2/Utils/Timer.h>
#include "Detectron2/evaluate/AveragePrecision.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// constants
static const char *WINDOW_NAME = "COCO detections";

CfgNode VisualizationDemo::setup_cfg(const std::string &config_file, const CfgNode::OptionList &opts,
	float confidence_threshold) {
	// load config from file and command-line arguments
	auto cfg = CfgNode::get_cfg();
	cfg.merge_from_file(config_file);
	cfg.merge_from_list(opts);
	// Set score_threshold for builtin models
	cfg["MODEL.RETINANET.SCORE_THRESH_TEST"] = confidence_threshold;
	cfg["MODEL.ROI_HEADS.SCORE_THRESH_TEST"] = confidence_threshold;
	cfg["MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH"] = confidence_threshold;
	cfg.freeze();
	return cfg;
}

void VisualizationDemo::start_val(const Options& options) {
	auto cfg = setup_cfg(options.config_file, options.opts, options.confidence_threshold);
	BuiltinDataset::register_all();
	VisualizationDemo demo(cfg);

	string train_path = "F:\\data\\faster_rcnn\\images\\train\\";
	string train_json = "F:\\data\\faster_rcnn\\annotations\\train.json";
	std::shared_ptr<CocoOriginalDataset> _dataset = std::make_shared<CocoOriginalDataset>(train_path, train_json);
	auto loader_opts = torch::data::DataLoaderOptions().batch_size(1).workers(1);
	auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
		(std::move(*_dataset), loader_opts);

	int tp = 0;
	int fp = 0;
	int fn = 0;
	std::vector<float> precisions;
	std::vector<float> recalls;
	for (auto& img_datas : *dataloader) {
		// check if lr needs to be warmed up at the begining
		std::vector<DatasetMapperOutput> inputs(1);
		if (img_datas.size() == 1)
		{
			std::cout << img_datas[0].file_name << std::endl;
			string path = img_datas[0].img_dir + img_datas[0].file_name;
			auto img = read_image(path, "BGR");
			int img_data_i = 0;
			//LoadData(img_datas, inputs, img_data_i);
			auto gt_boxes = img_datas[0].gt_bboxes;
			auto gt_labels = img_datas[0].gt_labels;
			auto device = gt_boxes.device();
			std::cout << "gt_boxes: " << gt_boxes << std::endl;
			std::cout << "gt_labels: " << gt_labels << std::endl;

			InstancesPtr predictions; VisImage visualized_output;
			tie(predictions, visualized_output) = demo.run_on_image(img);
			auto instances = dynamic_pointer_cast<Instances>(predictions->get("instances"));
			auto pred_boxes = instances->getTensor("pred_boxes");
			auto scores = instances->getTensor("scores");
			auto pred_classes = instances->getTensor("pred_classes");
			std::cout << "pred_boxes: " << pred_boxes << std::endl;
			std::cout << "pred_classes: " << pred_classes << std::endl;
			std::cout << "scores: " << scores << std::endl;

			int num_classes = 1;
			float pr_scale = 0.5;
			float overlap_threshold = 0.25;
			auto ap = Accumulators(num_classes, pr_scale, overlap_threshold);
			auto accumulator = ap.evaluate(pred_boxes.to(device), pred_classes.to(device), scores.to(device), gt_boxes, gt_labels);

			tp += accumulator[0].TP();
			fp += accumulator[0].FP();
			fn += accumulator[0].FN();
			auto pr = Precision(tp, fp, fn);
			auto recall = Recall(tp, fp, fn);
			precisions.push_back(pr);
			recalls.push_back(recall);

			//std::cout << "tp: " << tp << " fp: " << fp << " fn: " << fn << std::endl;
			std::cout << "pr: " << pr << " recall: " << recall << std::endl;
		}
	}
	std::cout << "tp: " << tp << " fp: " << fp << " fn: " << fn << std::endl;
	auto precision = Precision(tp, fp, fn);
	auto recall = Recall(tp, fp, fn);
	std::cout << "precision: " << precision << " recall: " << recall << std::endl;
	auto ap = compute_ap(precisions, recalls);
	std::cout<<"ap:"<<ap<<std::endl;


	//if (!options.input.empty()) {
	//	for (auto path : options.input) {
	//		// use PIL, to be consistent with evaluation
	//		auto img = read_image(path, "BGR");
	//		InstancesPtr predictions; VisImage visualized_output;
	//		//~!start_time = time.time()
	//		tie(predictions, visualized_output) = demo.run_on_image(img);
	//		
	//		if (!options.output.empty()) {
	//			string out_filename;
	//			if (File::IsDir(options.output)) {
	//				out_filename = File::ComposeFilename(options.output, File::Basename(path));
	//			}
	//			else {
	//				assert(options.input.size() == 1); // Please specify a directory with args.output
	//				out_filename = options.output;
	//			}
	//			visualized_output.save(out_filename);
	//		}
	//		else {
	//			cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
	//			cv::imshow(WINDOW_NAME, image_to_mat(torch::flip(visualized_output.get_image(), { -1 })));
	//			if (cv::waitKey(0) == 27) {
	//				break;  // esc to quit
	//			}
	//		}
	//	}
	//}
}


void VisualizationDemo::start(const Options &options) {
	auto cfg = setup_cfg(options.config_file, options.opts, options.confidence_threshold);
	BuiltinDataset::register_all();
	VisualizationDemo demo(cfg);

	if (!options.input.empty()) {
		for (auto path : options.input) {
			// use PIL, to be consistent with evaluation
			auto img = read_image(path, "BGR");
			InstancesPtr predictions; VisImage visualized_output;
            //~!start_time = time.time()
			tie(predictions, visualized_output) = demo.run_on_image(img);
			/*~!
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
			*/

			if (!options.output.empty()) {
				string out_filename;
				if (File::IsDir(options.output)) {
					out_filename = File::ComposeFilename(options.output, File::Basename(path));
				}
				else {
					assert(options.input.size() == 1); // Please specify a directory with args.output
					out_filename = options.output;
				}
				visualized_output.save(out_filename);
			}
			else {
				cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
				cv::imshow(WINDOW_NAME, image_to_mat(torch::flip(visualized_output.get_image(), { -1 })));
				if (cv::waitKey(0) == 27) {
					break;  // esc to quit
				}
			}
		}
	}
	else if (options.webcam) {
		assert(options.input.empty()); // Cannot have both --input and --webcam!
		assert(options.output.empty()); // output not yet supported with --webcam!
		auto cam = cv::VideoCapture(0);
		demo.run_on_video(cam, [](cv::Mat vis){
			cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
			cv::imshow(WINDOW_NAME, vis);
			return cv::waitKey(1) != 27; // esc to quit
		});
		cam.release();
		cv::destroyAllWindows();
	}
	else if (!options.video_input.empty()) {
		auto video = cv::VideoCapture(options.video_input);
		auto width = int(video.get(cv::CAP_PROP_FRAME_WIDTH));
		auto height = int(video.get(cv::CAP_PROP_FRAME_HEIGHT));
		auto frames_per_second = video.get(cv::CAP_PROP_FPS);
		auto num_frames = int(video.get(cv::CAP_PROP_FRAME_COUNT));
		auto filename = File::Basename(options.video_input);

		cv::VideoWriter output_file;
		if (!options.output.empty()) {
			string output_fname;
			if (File::IsDir(options.output)) {
				output_fname = File::ComposeFilename(options.output, filename);
				output_fname = File::ReplaceExtension(output_fname, ".mkv");
			}
			else {
				output_fname = options.output;
			}
			assert(!File::IsFile(output_fname));
			output_file = cv::VideoWriter(output_fname,
				// some installation of opencv may not support x264 (due to its license),
				// you can try other format (e.g. MPEG)
				cv::VideoWriter::fourcc('x', '2', '6', '4'), frames_per_second, { width, height }, true);
		}
		assert(File::IsFile(options.video_input));
		demo.run_on_video(video, [&](cv::Mat vis_frame){
			if (!options.output.empty()) {
				output_file.write(vis_frame);
				return true;
			}
			else {
				cv::namedWindow(filename, cv::WINDOW_NORMAL);
				cv::imshow(filename, vis_frame);
				return cv::waitKey(1) != 27; // esc to quit
			}
		});
		video.release();
		if (!options.output.empty()) {
			output_file.release();
		}
		else {
			cv::destroyAllWindows();
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VisualizationDemo::VisualizationDemo(const CfgNode &cfg, ColorMode instance_mode, bool parallel) :
	m_cpu_device(torch::kCPU), m_instance_mode(instance_mode), m_parallel(parallel)
{
	auto name = CfgNode::parseTuple<string>(cfg["DATASETS.TEST"], { "__unused" })[0];
	m_metadata = MetadataCatalog::get(name);
	if (parallel) {
		int num_gpu = torch::cuda::device_count();
		m_predictor = make_shared<AsyncPredictor>(cfg, num_gpu);
	}
	else {
		m_predictor = make_shared<DefaultPredictor>(cfg);
	}
}

std::tuple<InstancesPtr, VisImage> VisualizationDemo::run_on_image(torch::Tensor image) {
	VisImage vis_output;
	InstancesPtr predictions = m_predictor->predict(image);

	// Convert image from OpenCV BGR format to Matplotlib RGB format.
	image = torch::flip(image, { -1 });

	Visualizer visualizer(image, m_metadata, 1.0, m_instance_mode);
	if (predictions->has("panoptic_seg")) {
		auto panoptic_seg = dynamic_pointer_cast<PanopticSegment>(predictions->get("panoptic_seg"));
		vis_output = visualizer.draw_panoptic_seg_predictions(
			panoptic_seg->seg.to(m_cpu_device), panoptic_seg->infos);
	}
	else {
		if (predictions->has("sem_seg")) {
			vis_output = visualizer.draw_sem_seg(
				predictions->getTensor("sem_seg").argmax(0).to(m_cpu_device)
			);
		}
		if (predictions->has("instances")) {
			auto instances = dynamic_pointer_cast<Instances>(predictions->get("instances"));
			instances->to(m_cpu_device);
			vis_output = visualizer.draw_instance_predictions(instances);
		}
	}

	return { predictions, vis_output };
}

void VisualizationDemo::run_on_video(cv::VideoCapture &video, function<bool(cv::Mat)> vis_frame_processor) {
	VideoVisualizer video_visualizer(m_metadata, m_instance_mode);

	auto process_predictions = [&](cv::Mat &frame, InstancesPtr predictions,
		function<bool(cv::Mat)> vis_frame_processor){
		Timer timer("process_predictions");

		cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
		VisImage vis_frame;
		if (predictions->has("panoptic_seg")) {
			auto panoptic_seg = dynamic_pointer_cast<PanopticSegment>(predictions->get("panoptic_seg"));
			vis_frame = video_visualizer.draw_panoptic_seg_predictions(
				frame, panoptic_seg->seg.to(m_cpu_device), panoptic_seg->infos
			);
		}
		else if (predictions->has("instances")) {
			auto instances = dynamic_pointer_cast<Instances>(predictions->get("instances"));
			instances->to(m_cpu_device);
			vis_frame = video_visualizer.draw_instance_predictions(frame, instances);
		}
		else if (predictions->has("sem_seg")) {
			vis_frame = video_visualizer.draw_sem_seg(
				frame,
				predictions->getTensor("sem_seg").argmax(0).to(m_cpu_device)
			);
		}

		// Converts Matplotlib RGB format to OpenCV BGR format
		cv::Mat vis_frame_out;
		cv::cvtColor(image_to_mat(vis_frame.get_image()), vis_frame_out, cv::COLOR_RGB2BGR);
		return vis_frame_processor(vis_frame_out);
	};

	while (video.isOpened()) {
		cv::Mat frame;
		if (!video.read(frame) ||
			!process_predictions(frame, m_predictor->predict(image_to_tensor(frame)), vis_frame_processor)) {
			break;
		}
	}
}

void VisualizationDemo::analyze_on_video(cv::VideoCapture &video, VideoAnalyzer &analyzer) {
	auto process_predictions = [&](cv::Mat &frame, InstancesPtr predictions) {
			Timer timer("analyze_predictions");

			if (predictions->has("panoptic_seg")) {
				auto panoptic_seg = dynamic_pointer_cast<PanopticSegment>(predictions->get("panoptic_seg"));
				analyzer.on_panoptic_seg_predictions(
					frame, panoptic_seg->seg.to(m_cpu_device), panoptic_seg->infos
				);
			}
			else if (predictions->has("instances")) {
				auto instances = dynamic_pointer_cast<Instances>(predictions->get("instances"));
				instances->to(m_cpu_device);
				analyzer.on_instance_predictions(frame, instances, m_metadata->keypoint_names);
			}
			else if (predictions->has("sem_seg")) {
				auto sem_seq = predictions->getTensor("sem_seg").argmax(0).to(m_cpu_device);
				analyzer.on_sem_seg(frame, sem_seq);
			}
	};

	while (video.isOpened()) {
		cv::Mat frame;
		if (!video.read(frame)) {
			break;
		}
		process_predictions(frame, m_predictor->predict(image_to_tensor(frame)));
	}
}