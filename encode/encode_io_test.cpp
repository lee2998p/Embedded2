#include <opencv2/opencv.hpp>
#include "compressed_io.h"

#define IMG_HEIGHT 480
#define IMG_WIDTH 852

int main(int argc, char **argv) {
    cv::VideoCapture cap(0);
    cv::Mat frame, rgba;

    image_info info;
    info.height = IMG_HEIGHT;
    info.width = IMG_WIDTH;
    info.depth = 4;

    CompressedWriter writer("test", info, NVPIPE_HEVC);

    while (cap.isOpened()) {
        cap >> frame;

        cv::resize(frame, frame, cv::Size(IMG_WIDTH, IMG_HEIGHT));
        cv::cvtColor(frame, rgba, cv::COLOR_BGR2RGBA);
        std::vector<uint8_t> dataVec;
        if (rgba.isContinuous()) {
            dataVec.assign((uint8_t *) rgba.datastart, (uint8_t *) rgba.dataend);
            std::cout << "Its continuous" << std::endl;
        } else {
            std::cout << "Frame isn't continuous" << std::endl;
            for (int i = 0; i < rgba.rows; i++) {
                dataVec.insert(dataVec.end(), rgba.ptr<uint8_t>(i), rgba.ptr<uint8_t>(i) + rgba.cols);
            }
        }
        writer.write(dataVec);
        cv::imshow("Frame", frame);
        if (cv::waitKey(1) == 27)
            break;
    }

}
