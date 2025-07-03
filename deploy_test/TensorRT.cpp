#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>

using namespace nvinfer1;

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};
void test();
void deploy();
std::vector<char> readEngineFile(const std::string& enginePath) ;

int main() {
    // test();
    deploy();

    return 0;
}
void test(){
    std::cout << "TensorRT version: "
                << NV_TENSORRT_MAJOR << "."
                << NV_TENSORRT_MINOR << "."
                << NV_TENSORRT_PATCH << std::endl;

    Logger logger;
    auto builder = nvinfer1::createInferBuilder(logger);
    if (!builder) {
        std::cerr << "Failed to create builder" << std::endl;
        return;
    }


    std::cout << "Supports FP16: " << std::boolalpha
              << builder->platformHasFastFp16() << std::endl;

    delete builder;  // ✅ 正确释放
}
void deploy() {
    std::string enginePath = "log/DistgSSR_2xSR_5x5.engine";
    std::string imagePath = "input/HCI_new_bedroom/all.png";

    Logger logger;

    // Load engine
    auto engineData = readEngineFile(enginePath);
    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(logger));
    auto engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());

    // Input info
    const char* inputName = nullptr;
    const char* outputName = nullptr;

    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        TensorIOMode mode = engine->getTensorIOMode(name);
        if (mode == TensorIOMode::kINPUT) inputName = name;
        else if (mode == TensorIOMode::kOUTPUT) outputName = name;
    }

    // Read and preprocess image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    image.convertTo(image, CV_32F, 1.0 / 255.0);
    
    // Allocate device memory
    float* d_input = nullptr;
    float* d_output = nullptr;

    Dims inputDims = engine->getTensorShape(inputName);
    Dims outputDims = engine->getTensorShape(outputName);

    size_t inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; ++i) inputSize *= inputDims.d[i];
    size_t outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i) outputSize *= outputDims.d[i];

    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_input, image.ptr<float>(), inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Set tensor address
    context->setInputTensorAddress(inputName, d_input);
    context->setOutputTensorAddress(outputName, d_output);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    // Copy output
    std::vector<float> output(outputSize);
    cudaMemcpy(output.data(), d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "耗时: " << ms << " ms" << std::endl;

    // std::cout << "Inference done. First 10 values:\n";
    // for (int i = 0; i < std::min(size_t(10), outputSize); ++i) std::cout << output[i] << " ";
    // std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    // Display
    cv::Mat result(static_cast<int>(outputDims.d[2]), static_cast<int>(outputDims.d[3]), CV_32F, output.data());
    result.convertTo(result, CV_8U, 255.0);

    cv::imshow("image", image);
    cv::imshow("result", result);
    cv::waitKey(0);
}
std::vector<char> readEngineFile(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file) throw std::runtime_error("Engine file open failed: " + enginePath);

    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    return engineData;
}