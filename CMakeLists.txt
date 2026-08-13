cmake_minimum_required(VERSION 3.20)

project(
    llm_inference_accelerator
    VERSION 0.7.0
    LANGUAGES C CXX
)

include(CTest)
add_subdirectory(native)
