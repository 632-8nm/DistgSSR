#include <chrono>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <ratio>
#include <string>

void test();
cv::Mat pre_process();
void deploy();

int main() {
	// test();
	deploy();

	return 0;
}

void test() {
	// 创建环境对象
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "version_info");

	// 打印 ONNX Runtime 版本号
	std::cout << "ONNX Runtime 版本号: " << Ort::GetVersionString()
			  << std::endl;

	// 检查是否支持 CUDA / CPU（取决于构建的 runtime）
	auto start = std::chrono::high_resolution_clock::now();
	Ort::SessionOptions session_options;
	Ort::Session session(env, "model.onnx", session_options);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;

	std::cout << "模型成功加载：" << session.GetInputCount() << " 个输入；"
			  << session.GetOutputCount() << " 个输出。" << std::endl;
	std::cout << "耗时 " << duration.count() << " ms" << std::endl;
}
cv::Mat pre_process() {
	std::string input_path = "input/HCI_new_bedroom/";
	cv::Mat temp =
		cv::imread(input_path + "view_03_03.png", cv::IMREAD_GRAYSCALE);
	cv::Mat processed_image(temp.size() * 5, temp.type());
	for (int i = 0; i < 25; i++) {
		int row = i / 5;
		int col = i % 5;
		std::string filename = input_path + "view_0" + std::to_string(row + 1)
							   + "_0" + std::to_string(col + 1) + ".png";
		// std::cout << filename << std::endl;
		cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
		image.copyTo(processed_image(
			cv::Range(row * image.rows, (row + 1) * image.rows),
			cv::Range(col * image.cols, (col + 1) * image.cols)));
	}
	// cv::imshow("1", processed_image);
	// cv::imwrite(input_path + "all.png", processed_image);
	// cv::waitKey(1000);
	return processed_image;
}
void deploy() {
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "version_info");

	Ort::SessionOptions session_options;
	session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
	session_options.DisableMemPattern();
	// session_options.SetIntraOpNumThreads(1); // 单线程以避免线程池初始化干扰
	auto start = std::chrono::high_resolution_clock::now();
	Ort::Session session(env, "log/DistgSSR_2xSR_5x5.onnx", session_options);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;

	std::cout << "模型成功加载：" << session.GetInputCount() << " 个输入；"
			  << session.GetOutputCount() << " 个输出。" << std::endl;
	std::cout << "加载耗时: " << duration.count() << " ms" << std::endl;

	// 读取图像

	cv::Mat image = pre_process();
	image.convertTo(image, CV_32FC(image.channels()), 1.0 / 255.0);
	std::vector<float> input_data(image.begin<float>(), image.end<float>());
	std::vector<int64_t> input_shape = {1, 1, 640, 640};

	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
		OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		memory_info, input_data.data(), input_data.size(), input_shape.data(),
		input_shape.size());

	// 获取输入输出名
	Ort::AllocatorWithDefaultOptions allocator;

	Ort::AllocatedStringPtr input_name_ptr =
		session.GetInputNameAllocated(0, allocator);
	const char* input_name = input_name_ptr.get();

	Ort::AllocatedStringPtr output_name_ptr =
		session.GetOutputNameAllocated(0, allocator);
	const char* output_name = output_name_ptr.get();

	// 推理
	std::vector<const char*> input_names{input_name};
	std::vector<const char*> output_names{output_name};

	start = std::chrono::high_resolution_clock::now();
	auto output_tensors =
		session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor,
					1, output_names.data(), 1);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	std::cout << "推理耗时: " << duration.count() << " ms" << std::endl;
	// 提取输出张量
	Ort::Value& output_tensor = output_tensors[0];
	float* output_data = output_tensor.GetTensorMutableData<float>();

	// 获取输出形状
	auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
	if (output_shape.size() != 4 || output_shape[0] != 1
		|| output_shape[1] != 1) {
		std::cerr << "输出不是 [1,1,H,W] 形状，实际是：";
		for (auto s : output_shape) std::cerr << s << " ";
		std::cerr << std::endl;
		return;
	}

	int out_h = static_cast<int>(output_shape[2]);
	int out_w = static_cast<int>(output_shape[3]);

	// 构造 cv::Mat，注意需要复制数据
	cv::Mat output_image(out_h, out_w, CV_32F, output_data);

	// 可选：归一化到 [0,255] 显示范围
	cv::Mat display_image;
	cv::normalize(output_image, display_image, 0, 255, cv::NORM_MINMAX);
	display_image.convertTo(display_image, CV_8U);
	cv::imshow("before", image);
	cv::imshow("after", display_image);
	cv::imwrite("upsampled.png", display_image);
	std::cout << display_image.size << std::endl;
	// cv::waitKey();

	std::cout << "✅ 推理完成！" << std::endl;
}