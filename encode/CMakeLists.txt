cmake_minimum_required(VERSION 3.10)
project(encoding)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

set(NVPIPE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(NvPipe)

set(LIBS ${OpenCV_LIBS} NvPipe)

include_directories(${OpenCV_INCLUDE_DIRS} NvPipe/build/include/)

add_executable(client_nvenc client_nvenc.cpp)
target_link_libraries(client_nvenc ${LIBS})
add_executable(server_nvenc server_nvenc.cpp)
target_link_libraries(server_nvenc ${LIBS})


add_library(compressed_io SHARED compressed_io.cpp)
target_link_libraries(compressed_io NvPipe)

add_executable(encode_test encode_io_test.cpp)
add_executable(decode_test decode_io_test.cpp)
target_link_libraries(encode_test ${LIBS} compressed_io)
target_link_libraries(decode_test ${LIBS} compressed_io)

add_executable(encode_compare encode_compare.cpp)
target_link_libraries(encode_compare ${LIBS} compressed_io)
