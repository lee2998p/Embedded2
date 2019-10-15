#include <iostream>
#include <vector>
#include "compressed_io.h"


CompressedWriter::CompressedWriter(std::string base_name, image_info frame_info, NvPipe_Codec codec,
                                   NvPipe_Compression compression, NvPipe_Format format,
                                   uint64_t bitrate, uint32_t frame_rate) {

    extension = codec == NVPIPE_H264 ? ".h264" : ".hevc";
    filename = base_name + extension;
    file = std::ofstream(filename, std::ios::binary | std::ios::out);
    info = frame_info;
    file.write((char *) &info, sizeof(image_info));

    encoder = NvPipe_CreateEncoder(format, codec, compression, bitrate, frame_rate,
                                   info.width, info.width);
    if (!encoder) {
        std::cerr << "Failed to create encoder" << std::endl;
    }

    dataPitch = info.width * 4;


}

CompressedWriter::~CompressedWriter() {
    file.close();
    NvPipe_Destroy(encoder);
}

void CompressedWriter::write(std::vector<uint8_t> frameData) {
    std::vector<uint8_t> compressed(frameData.size());
    auto start = std::chrono::high_resolution_clock::now();
    uint64_t size = NvPipe_Encode(encoder, frameData.data(), info.width * 4, compressed.data(), compressed.size(),
                                  info.width, info.height, false);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "codec is {" << extension << "} encode time is {" << duration.count() << "} us compressed data is {"
              << ((float) size / frameData.size()) * 100 << "}% of original frame" << std::endl;
    file.write((char *) &size, sizeof(uint64_t));
    file.write((char *) compressed.data(), size);

}


CompressedReader::CompressedReader(std::string base_name, NvPipe_Codec codec, NvPipe_Format format) {
    extension = codec == NVPIPE_H264 ? ".h264" : ".hevc";
    filename = base_name + extension;
    file = std::ifstream(filename, std::ios::in | std::ios::binary);
    file.read((char *) &info, sizeof(image_info));

    decoder = NvPipe_CreateDecoder(format, codec, info.width, info.height);
    dataSize = info.height * info.width * 4;
}

bool CompressedReader::read(std::vector<uint8_t> &frame) {
    uint64_t size;
    file.read((char *) &size, sizeof(uint64_t));
    std::vector<uint8_t> compressed(info.height * info.width * 4);
    file.read((char *) compressed.data(), size);
    frame.clear();
    frame.reserve(info.width * info.height * 4);
    uint64_t ret = NvPipe_Decode(decoder, compressed.data(), size, frame.data(), info.width, info.height);
    if (ret == size)
        std::cerr << "Decode error" << std::endl;
    return true;
}

CompressedReader::~CompressedReader() {
    NvPipe_Destroy(decoder);
    file.close();
}

image_info CompressedReader::fileInfo() {
    return info;
}
