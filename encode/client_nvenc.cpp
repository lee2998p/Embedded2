#include <iostream>
#include <NvPipe.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define SERVER_IP "127.0.0.1"
#define SERVER_PORT 9001
#define IMG_HEIGHT 200
#define IMG_WIDTH 200

// Creates and connects a socket
void setup_client_socket(char *addr, int port, int &sock, sockaddr_in &serv_addr) {
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("Failed to create socket\n");
        exit(EXIT_FAILURE);
    }
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, addr, &serv_addr.sin_addr) <= 0) {
        printf("[!!] Address invalid/not supported\n");
        exit(EXIT_FAILURE);
    }

    if (connect(sock, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        printf("Failed to connect to server\n");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    cv::VideoCapture cam(0);
    cv::Mat frame;
    uint32_t height = IMG_HEIGHT, width = IMG_WIDTH;
    NvPipe *encoder = NvPipe_CreateEncoder(NVPIPE_RGBA32, NVPIPE_H264, NVPIPE_LOSSLESS, 32 * 1000 * 1000,
                                           cam.get(cv::CAP_PROP_FPS), width, height);

    int sock = 0;
    sockaddr_in serv_addr;
    setup_client_socket(SERVER_IP, SERVER_PORT, sock, serv_addr);

    std::vector<uint8_t> compressed(width * height * 4);
    while (cam.isOpened()) {
        cam >> frame;
        cv::resize(frame, frame, cv::Size(IMG_WIDTH, IMG_HEIGHT));
        cv::Mat rgbaframe(width, height, CV_8UC4);
        cv::cvtColor(frame, rgbaframe, cv::COLOR_BGR2RGBA);

        uint64_t compressedSize = NvPipe_Encode(encoder, rgbaframe.data, width * 4, compressed.data(),
                                                compressed.size(), width, height, true);
        printf("Raw size: %ld\nEncoded size: %ld\n\n", frame.cols * frame.rows * 4 * sizeof(char), compressedSize);
        send(sock, &compressedSize, sizeof(uint64_t), 0);
        send(sock, compressed.data(), compressedSize, 0);
    }
    NvPipe_Destroy(encoder);
    exit(EXIT_SUCCESS);
}
