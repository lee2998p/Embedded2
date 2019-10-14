#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <NvPipe.h>

// TODO fix issue of rebuilding from more than 1 packet
#define LISTEN_PORT 9001
#define IMG_HEIGHT 200
#define IMG_WIDTH 200

void setup_server_socket(int port, int &sock, sockaddr_in &address, int opt) {
    int server_fd;
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        printf("[!!] Error: unable to start socket\n");
        exit(EXIT_FAILURE);
    }
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        printf("[!!] Error: setsockopt failed\n");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(LISTEN_PORT);

    if (bind(server_fd, (sockaddr *) &address, sizeof(address)) < 0) {
        printf("[!!] Error: unable to bind\n");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0) {
        printf("[!!] Error: failed to listen\n");
        exit(EXIT_FAILURE);
    }
    int addrlen = sizeof(address);
    if ((sock = accept(server_fd, (sockaddr *) &address, (socklen_t *) &addrlen)) < 0) {
        printf("[!!] Error: failed to accept connection\n");
        exit(EXIT_FAILURE);
    }
}


int main(int argc, char **argv) {
    int sock;
    sockaddr_in addr;
    int opt = 1;
    setup_server_socket(LISTEN_PORT, sock, addr, opt);
    printf("Got connection\n");

    uint32_t width = IMG_WIDTH, height = IMG_HEIGHT;
    // Buffer to store image
    std::vector<uint8_t> compressed_frame(width * height * 4);
    // Create decoder
    NvPipe *decoder = NvPipe_CreateDecoder(NVPIPE_RGBA32, NVPIPE_H264, width, height);

    while (true) {
        uint64_t compressedSize;
        read(sock, &compressedSize, sizeof(uint64_t));
        read(sock, compressed_frame.data(), compressedSize);

        cv::Mat frame(width, height, CV_8UC4);
        NvPipe_Decode(decoder, compressed_frame.data(), compressedSize, frame.data, width, height);
        uint64_t raw_size = frame.cols * frame.rows * 4 * sizeof(uint8_t);
        printf("Frame info:\n\tRaw size: %ld\n\tEncoded size: %ld\n\tEncode ratio: %f\%\n\n", raw_size, compressedSize, ((float)compressedSize/raw_size) * 100);

        // Convert back for display
        cv::cvtColor(frame, frame, cv::COLOR_RGBA2BGR);
        cv::imshow("FRAME", frame);

        if (cv::waitKey(1) == 27) {
            printf("ESC pressed, closing\n");
            break;
        }
    }
    NvPipe_Destroy(decoder);
    exit(EXIT_SUCCESS);
}
