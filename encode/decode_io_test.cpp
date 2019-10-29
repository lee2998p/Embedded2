#include <opencv2/opencv.hpp>
#include "compressed_io.h"

int main(int argc, char **argv) {

    if(argc < 2){
      std::cout << "Usage: " << argv[0] << " <filename_no_ext>" << std::endl;
    }
    cv::Mat frame;
    CompressedReader reader(argv[1], NVPIPE_HEVC);

    image_info info = reader.fileInfo();
    cv::Mat rgba(cv::Size(info.width, info.height), CV_8UC4);

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
