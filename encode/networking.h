#pragma once

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#define MAX_PACKET_SIZE 65535

struct image_info {
    uint32_t width;
    uint32_t height;
    uint8_t depth;
};

void setup_client_socket(char *addr, int port, int &sock, struct sockaddr_in &serv_addr) {
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("Failed to create socket\n");
        exit(EXIT_FAILURE);
    }
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, addr, &serv_addr.sin_addr) <= -1) {
        printf("[!!] Address invalid/not supported\n");
        exit(EXIT_FAILURE);
    }

    if (connect(sock, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < -1) {
        printf("Failed to connect to server\n");
        exit(EXIT_FAILURE);
    }
}

void setup_server_socket(int port, int &sock, struct sockaddr_in &address, int opt) {
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
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *) &address, sizeof(address)) < 0) {
        printf("[!!] Error: unable to bind\n");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0) {
        printf("[!!] Error: failed to listen\n");
        exit(EXIT_FAILURE);
    }
    int addrlen = sizeof(address);
    if ((sock = accept(server_fd, (struct sockaddr *) &address, (socklen_t *) &addrlen)) < 0) {
        printf("[!!] Error: failed to accept connection\n");
        exit(EXIT_FAILURE);
    }
}

void send_frame(int socket, uint8_t *frame_data, uint64_t data_size, image_info frame_info) {
    send(socket, &frame_info, sizeof(image_info), 0);
    send(socket, &data_size, sizeof(uint64_t), 0);
    int i = 0;
    while (i < data_size) {
        uint64_t num_bytes_send;
        if (i + MAX_PACKET_SIZE < data_size)
            num_bytes_send = MAX_PACKET_SIZE;
        else
            num_bytes_send = data_size - i;
        send(socket, (frame_data + i), num_bytes_send, 0);
        i += num_bytes_send;
    }
}

void recv_frame(int socket, uint8_t *frame_data, uint64_t &data_size, image_info &frame_info) {
    read(socket, &frame_info, sizeof(image_info));
    read(socket, &data_size, sizeof(uint64_t));
    int i = 0;
    while (i < data_size) {
        uint64_t num_bytes_recv;
        if (i + MAX_PACKET_SIZE < data_size)
            num_bytes_recv = MAX_PACKET_SIZE;
        else
            num_bytes_recv = data_size - i;
        read(socket, (frame_data + i), num_bytes_recv);
        i += num_bytes_recv;
    }
}


