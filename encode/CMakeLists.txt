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

