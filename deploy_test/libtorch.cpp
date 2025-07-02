#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <chrono>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/script.h> // One-stop header for loading TorchScript models
#include <torch/torch.h>

void		  test_print_info();
void		  test_deploy();
torch::Tensor load_data(const std::string& path, int angRes);
void		  deploy(const std::string& path, int angRes, torch::Device device);
void		  test_onnx();
cv::Mat		  load_lf_sai(const std::string& path, int angRes);

int main() {
	// test_print_info();
	// test_deploy();
	// std::string path   = "input/HCI_new_bedroom/";
	// int			angRes = 5;
	// deploy(path, angRes, torch::kCPU);
	// deploy(path, angRes, torch::kMPS); // slow

	return 0;
}
void deploy(const std::string& path, int angRes, torch::Device device) {
	torch::jit::script::Module module =
		torch::jit::load("log/DistgSSR_2xSR_5x5.pt");
	module.to(device);

	torch::Tensor input = load_data(path, angRes);

	input = input.unsqueeze(0).unsqueeze(0);
	input = input.to(device);
	std::cout << input.sizes() << std::endl;

	std::vector<torch::jit::IValue> inputs = {input};
	torch::Tensor					output;

	auto start = std::chrono::high_resolution_clock::now();
	try {
		output = module.forward(inputs)
					 .toTensor(); // 输出形状: [1, 1, u*h*scale, v*w*scale]
		output = output.squeeze(); // 移除 batch 和 channel 维度 -> [u*h*scale,
								   // v*w*scale]
	} catch (const c10::Error& e) {
		std::cerr << "推理失败: " << e.what() << std::endl;
		return;
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration =
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "推理时间: " << duration.count() << " 毫秒" << std::endl;

	// 5
	cv::Mat center = cv::imread(path + "view_03_03.png", cv::IMREAD_GRAYSCALE);

	int scale  = 2;			  // 超分倍数 (根据模型定义调整)
	int height = center.rows; // 原始高度
	int width  = center.cols; // 原始宽度

	output = output
				 .view({angRes, height * scale, angRes,
						width * scale})	 // [u, h*scale, v, w*scale]
				 .permute({0, 2, 1, 3}); // [u, v, h*scale, w*scale]

	std::cout << "输出张量形状: " << output.sizes() << std::endl;

	// 6
	torch::Tensor temp = output[2][2];
	if (device == torch::kMPS) {
		temp = temp.to(torch::kCPU); // MPS 设备需要转换到 CPU
	}
	temp = temp.mul(255).clamp(0, 255).to(torch::kU8);
	cv::Mat center_upsampled(temp.size(0), temp.size(1), CV_8U,
							 temp.data_ptr<uint8_t>());

	cv::imshow("center", center);
	cv::imshow("center_upsampled", center_upsampled);
	cv::waitKey();
}
torch::Tensor load_data(const std::string& path, int angRes) {
	cv::Mat first_image =
		cv::imread(path + "/view_01_01.png", cv::IMREAD_GRAYSCALE);
	int			  height = first_image.rows;
	int			  width	 = first_image.cols;
	torch::Tensor tensor =
		torch::zeros({angRes, angRes, height, width}, torch::kFloat32);

	for (int i = 0; i < angRes; i++) {
		for (int j = 0; j < angRes; j++) {
			std::string filename = path + "/view_0" + std::to_string(i + 1)
								   + "_0" + std::to_string(j + 1) + ".png";
			cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
			image.convertTo(image, CV_32F, 1.0 / 255.0);
			if (image.empty()) {
				std::cerr << "无法加载图像: " << filename << std::endl;
				continue;
			}
			tensor[i][j] =
				torch::from_blob(image.data, {height, width}, torch::kFloat32)
					.clone();
		}
	}
	torch::Tensor result = tensor.permute({0, 2, 1, 3}).contiguous();
	return result.view({result.size(0) * result.size(1),   // u * h
						result.size(2) * result.size(3)}); // v * w
}

void test_deploy() {
	torch::jit::script::Module module;
	std::cout << "当前工作目录: " << std::filesystem::current_path()
			  << std::endl;

	try {
		module = torch::jit::load("log/DistgSSR_2xSR_5x5.pt");
		module.to(torch::kCPU);
	} catch (const c10::Error& e) {
		std::cerr << "加载模型失败: " << e.what() << std::endl;
		return;
	}
	std::vector<torch::jit::IValue> inputs;

	torch::Tensor input = torch::ones({1, 1, 640, 640});
	input				= input.to(torch::kCPU);
	inputs.push_back(input); // 示例输入

	// 执行推理
	auto		  start	 = std::chrono::high_resolution_clock::now();
	torch::Tensor output = module.forward(inputs).toTensor();
	auto		  end	 = std::chrono::high_resolution_clock::now();
	auto		  duration =
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "推理时间: " << duration.count() << " 毫秒" << std::endl;

	std::cout << inputs[0].toTensor().sizes() << std::endl;
	std::cout << output.sizes() << std::endl;
}
void test_print_info() {
	// 1. 输出 OpenCV 信息
	std::cout << "=== OpenCV 信息 ===" << std::endl;
	std::cout << "版本: " << CV_VERSION << std::endl;

	// 2. 输出 LibTorch 信息
	std::cout << "\n=== LibTorch 信息 ===" << std::endl;
	std::cout << "版本: " << TORCH_VERSION << std::endl;
	std::cout << "CUDA 可用: " << std::boolalpha << torch::cuda::is_available()
			  << std::endl;
	std::cout << "MPS 可用: " << torch::hasMPS() << std::endl; // macOS 专属
	std::cout << "CPU 线程数: " << torch::get_num_threads() << std::endl;

	// 3. 测试张量计算
	std::cout << "\n=== 测试计算 ===" << std::endl;
	at::Tensor x = torch::rand({2, 3});
	std::cout << "随机张量:\n" << x << std::endl;
	std::cout << "张量求和: " << x.sum() << std::endl;

	// 4. 测试 OpenCV 图像处理
	std::cout << "\n=== 测试图像处理 ===" << std::endl;
	cv::Mat img(100, 100, CV_8UC3, cv::Scalar(100, 255, 0));
	std::cout << "图像尺寸: " << img.size() << std::endl;
	cv::imshow("Green Image", img);
	cv::waitKey(0); // 等待按键
}
cv::Mat load_lf_sai(const std::string& path, int angRes) {
	cv::Mat full_img;

	for (int u = 0; u < angRes; ++u) {
		cv::Mat row_img;
		for (int v = 0; v < angRes; ++v) {
			char filename[256];
			snprintf(filename, sizeof(filename), "%s/view_0%d_0%d.png",
					 path.c_str(), u + 1, v + 1);

			cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
			if (img.empty()) {
				std::cerr << "无法读取图像: " << filename << std::endl;
				exit(1);
			}
			img.convertTo(img, CV_32FC1, 1.0 / 255.0);
			if (row_img.empty())
				row_img = img;
			else
				cv::hconcat(row_img, img, row_img);
		}
		if (full_img.empty())
			full_img = row_img;
		else
			cv::vconcat(full_img, row_img, full_img);
	}
	return full_img;
}
