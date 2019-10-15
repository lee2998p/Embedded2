#include <opencv2/opencv.hpp>
#include "compressed_io.h"

typedef struct {
    uint32_t width;
    uint32_t height;
} dim;

int main(int argc, char **argv) {
//    std::ifstream file("/home/jrmo/compression_data/akiyo.bgr", std::ios::binary | std::ios::in);
//    std::ifstream file("/home/jrmo/compression_data/bridge_close.bgr", std::ios::binary | std::ios::in);
    std::ifstream file("/home/jrmo/compression_data/bridge_far.bgr", std::ios::binary | std::ios::in);
    file.seekg(0, std::ios::end);
    uint64_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    dim d;
    file.read((char *) &d, sizeof(dim));
    std::vector<uint8_t> bgrData(d.width * d.height * 3);
    cv::Mat frame(d.height, d.width, CV_8UC3);
    cv::Mat rgba(d.height, d.width, CV_8UC4);
    uint32_t size;

    image_info info;
    info.depth = 4;
    info.width = d.width;
    info.height = d.height;

    std::string name = "bridge_far";
    CompressedWriter writerHEVC(name, info, NVPIPE_HEVC, NVPIPE_LOSSLESS, NVPIPE_RGBA32, 32 * 1000 * 1000);
    CompressedWriter writerh264(name, info, NVPIPE_H264, NVPIPE_LOSSLESS, NVPIPE_RGBA32, 32 * 1000 * 1000);
    while (file.tellg() != file_size) {
        file.read((char *) &size, sizeof(uint32_t));
        file.read((char *) bgrData.data(), size);
        frame.data = bgrData.data();
        cv::cvtColor(frame, rgba, cv::COLOR_BGR2RGBA);
        std::vector<uint8_t> dataVec;
        if (rgba.isContinuous())
            dataVec.assign((uint8_t *) rgba.datastart, (uint8_t *) rgba.dataend);
        writerHEVC.write(dataVec);
        writerh264.write(dataVec);
    }


}