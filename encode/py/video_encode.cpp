#include "video_encode.h"


VideoEncoder::VideoEncoder(std::string base_name, uint32_t width, uint32_t height,
                          NvPipe_Codec codec, NvPipe_Compression compression, NvPipe_Format format,
                          uint32_t frame_rate, uint64_t bitrate){

    extension = codec == NVPIPE_H264 ? ".h264" : ".hevc";
    filename = base_name + extension;
    file = std::ofstream(filename, std::ios::binary | std::ios::out);
    info.width = width;
    info.height = height;
    info.depth = 4; // For now force color


    printf("Info\n\twidth: %d\n\theight: %d\n\tdepth: %d\n", info.width, info.height, info.depth);
    file.write((char *) &info, sizeof(frame_info));

    encoder = NvPipe_CreateEncoder(format, codec, compression, bitrate, frame_rate,
                                   info.width, info.width);
    if (!encoder) {
        std::cerr << "Failed to create encoder" << std::endl;
    }

    dataPitch = info.width * 4;

}

// for now gotta convert outside of module
void VideoEncoder::write(torch::Tensor f){
  std::cout << "Writing tensor with shape " << f.sizes() << std::endl;
  if(f.sizes()[2] == 3){
    printf("Adding alpha channel\n");
    // Add an extra channel for alpha
    torch::Tensor alpha = torch::full({f.sizes()[0], f.sizes()[1], 1}, 255, f.options());
    f = torch::cat({f, alpha}, 2);
  }
  // This should be right for accessing the tensor data raw
  uint8_t *data_ptr = f.data_ptr<uint8_t>();
  uint64_t size = f.sizes()[0] * f.sizes()[1] * f.sizes()[2];
  // We're gonna need these regardless of which device we're using
  uint8_t *compressedData = (uint8_t *) malloc(size);
  uint64_t compressedSize;

  bool cuda = f.device() == torch::kCUDA; // Figure out what device the tensor is on
  if(cuda){
    printf("Using device memory\n");
    uint8_t *compressedDevData;
    cudaMalloc((void **)&compressedDevData, size);
    compressedSize = NvPipe_Encode(encoder, data_ptr,
                                  info.width * info.depth,
                                  compressedDevData, size, info.width, info.height, false);
    // Can't just write to a file from device mem afaik
    cudaMemcpy(compressedData, compressedDevData, compressedSize, cudaMemcpyDeviceToHost);
    cudaFree(compressedDevData);
  } else {
    printf("Using host memory\n");
    compressedSize = NvPipe_Encode(encoder, data_ptr,
                                  info.width * info.depth,
                                  compressedData, size, info.width, info.height, false);
  }
  std::cout << "Writing " << compressedSize << " bytes to file" << std::endl;
  printf("Comresison saves %ld bytes -> compressed data is %f %% smaller\n\n", size - compressedSize, (float)compressedSize/size * 100);
  file.write((char *) &compressedSize, sizeof(uint64_t));
  file.write((char *) compressedData, compressedSize);
}


VideoEncoder::~VideoEncoder(){
  file.close();
  NvPipe_Destroy(encoder);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
py::class_<VideoEncoder>(m, "VideoEncoder")
  .def(py::init<std::string, uint32_t, uint32_t>())
  .def("write", &VideoEncoder::write);
}
