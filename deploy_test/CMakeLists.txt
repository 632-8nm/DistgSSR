cmake_minimum_required(VERSION 3.20)
project(deploy)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ================ libraries ================
set(OpenCV_DIR "D:\\Libraries\\opencv-4.11.0_cuda-12.8")
set(Torch_DIR "D:\\Libraries\\libtorch-win-shared-with-deps-debug-2.7.1+cu128")
# set(Torch_DIR "D:\\Libraries\\libtorch-win-shared-with-deps-2.7.1+cu128")
set(ONNXRUNTIME_DIR "D:\\Libraries\\onnxruntime-win-x64-1.22.0")
set(TENSORRT_DIR "D:\\Libraries\\TensorRT-10.10.0.31")
# ================ libraries ================
set(CMAKE_PREFIX_PATH "${OpenCV_DIR};${Torch_DIR}")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs videoio)
find_package(CUDAToolkit REQUIRED)

add_executable(libtorch libtorch.cpp)
target_link_libraries(libtorch PRIVATE "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")
foreach(lib ${OpenCV_LIBS})
    add_custom_command(TARGET libtorch POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:${lib}>"
        "$<TARGET_FILE_DIR:libtorch>"
    )
endforeach()
add_custom_command(TARGET libtorch POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/log"
    "$<TARGET_FILE_DIR:libtorch>/log"
)
add_custom_command(TARGET libtorch POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/input"
    "$<TARGET_FILE_DIR:libtorch>/input"
)

add_executable(onnx onnx.cpp)
target_include_directories(onnx PRIVATE "${ONNXRUNTIME_DIR}/include")
target_link_directories(onnx PRIVATE "${ONNXRUNTIME_DIR}/lib")
target_link_libraries(onnx PRIVATE "${OpenCV_LIBS}" onnxruntime)

add_executable(TensorRT TensorRT.cpp)
target_include_directories(TensorRT PRIVATE
    "${TENSORRT_DIR}/include"
    "${OpenCV_DIR}/include"
)
target_link_libraries(TensorRT PRIVATE
    "${OpenCV_LIBS}"
    "${TENSORRT_DIR}/lib/nvinfer_10.lib"
    "${CUDAToolkit_LIBRARY_DIR}/cudart.lib"
)


message(STATUS "OpenCV 头文件 目录: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV 库文件目录: ${OpenCV_LIB_DIR}")
message(STATUS "OpenCV 库文件: ${OpenCV_LIBS}")
message(STATUS "OpenCV 版本: ${OpenCV_VERSION}")

message(STATUS "LibTorch 头文件目录: ${TORCH_INCLUDE_DIRS}")
message(STATUS "LibTorch 库目录: ${TORCH_LIBRARY_DIR}")
message(STATUS "LibTorch 库文件: ${TORCH_LIBRARIES}")
message(STATUS "LibTorch 版本: ${Torch_VERSION}")


