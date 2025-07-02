cmake_minimum_required(VERSION 3.20)
project(deploy)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ================ libraries ================
set(OpenCV_DIR "/usr/local/opencv/Release")
set(Torch_DIR "/usr/local/libtorch")
set(ONNXRUNTIME_DIR "/usr/local/onnxruntime-osx-arm64-1.22.0")
# ================ libraries ================
set(CMAKE_PREFIX_PATH "${OpenCV_DIR};${Torch_DIR}")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs dnn)

add_executable(libtorch libtorch.cpp)

target_link_libraries(libtorch PRIVATE "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")

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


message(STATUS "OpenCV 头文件目录: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV 库文件目录: ${OpenCV_LIB_DIR}")
message(STATUS "OpenCV 库文件: ${OpenCV_LIBS}")
message(STATUS "OpenCV 版本: ${OpenCV_VERSION}")

message(STATUS "LibTorch 头文件目录: ${TORCH_INCLUDE_DIRS}")
message(STATUS "LibTorch 库目录: ${TORCH_LIBRARY_DIR}")
message(STATUS "LibTorch 库文件: ${TORCH_LIBRARIES}")
message(STATUS "LibTorch 版本: ${Torch_VERSION}")


