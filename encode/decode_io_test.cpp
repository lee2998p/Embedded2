#include <opencv2/opencv.hpp>
#include "compressed_io.h"

#define IMG_HEIGHT 288
#define IMG_WIDTH 352

int main(int argc, char **argv) {
    cv::Mat frame;
    cv::Mat rgba(cv::Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC4);
    CompressedReader reader("compare", NVPIPE_HEVC);

    image_info info = reader.fileInfo();
    std::vector<uint8_t> frameData(1);
    while (true) {
        reader.read(frameData);
        rgba.data = frameData.data();
        cv::cvtColor(rgba, frame, cv::COLOR_RGBA2BGR);
        cv::imshow("Frame", frame);
        if (cv::waitKey(33) == 27) // Hard code the frame rate to 30 fps
            break;
    }

}
