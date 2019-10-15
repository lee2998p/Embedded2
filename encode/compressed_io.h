#pragma once

#include <NvPipe.h>
#include <fstream>
#include <chrono>

typedef struct {
    uint32_t width;
    uint32_t height;
    uint8_t depth;
} image_info;


class CompressedWriter {
public:
    CompressedWriter(std::string base_name, image_info frame_info, NvPipe_Codec codec = NVPIPE_H264,
                     NvPipe_Compression compression = NVPIPE_LOSSLESS, NvPipe_Format format = NVPIPE_RGBA32,
                     uint64_t bitrate = 32 * 1000 * 1000, uint32_t frame_rate = 30);

    ~CompressedWriter();

    void write(std::vector<uint8_t> frameData);

private:
    std::string extension, filename;
    image_info info;
    NvPipe *encoder;
    std::ofstream file;
    uint64_t dataPitch;
    bool forceIFrame;
};

class CompressedReader {
public:
    CompressedReader(std::string base_name,
                     NvPipe_Codec codec = NVPIPE_H264, NvPipe_Format format = NVPIPE_RGBA32);

    bool read(std::vector<uint8_t> &frame);

    ~CompressedReader();

    image_info fileInfo();

private:
    NvPipe *decoder;
    std::ifstream file;
    std::string filename, extension;
    image_info info;
    uint64_t dataSize;

};