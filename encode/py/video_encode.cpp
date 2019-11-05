#include "video_encode.h"

//#define DEBUG

#ifdef DEBUG
    #define PRINTDBG printf
#else
    #define PRINTDBG
#endif

VideoEncoder::VideoEncoder(std::string base_name, uint32_t width, uint32_t height,
                          NvPipe_Codec codec, NvPipe_Compression compression, NvPipe_Format format,
                          uint32_t frame_rate, uint64_t bitrate){

    extension = codec == NVPIPE_H264 ? ".h264" : ".hevc";
    filename = base_name + extension;
    file = std::ofstream(filename, std::ios::binary | std::ios::out | std::ios::trunc);
    info.width = width;
    info.height = height;
    info.depth = 4; // For now force color

    PRINTDBG("Info\n\twidth: %d\n\theight: %d\n\tdepth: %d\n", info.width, info.height, info.depth);

    file.write((char *) &info, sizeof(frame_info));

    encoder = NvPipe_CreateEncoder(format, codec, compression, bitrate, frame_rate,
                                   info.width, info.width);
    if (!encoder) {
        std::cerr << "Failed to create encoder" << std::endl;
    }

}

void VideoEncoder::write(torch::Tensor frame, torch::Tensor ROI, char *IV, uint32_t iv_size){
  if(frame.sizes()[2] == 3){
    PRINTDBG("Adding alpha channel\n");
    torch::Tensor alpha = torch::full({frame.sizes()[0], frame.sizes()[1], 1}, 255, frame.options());
    frame = torch::cat({frame, alpha}, 2);
  }

  // Grab the frame data_ptr
  uint8_t *data_ptr = frame.data_ptr<uint8_t>();
  uint64_t size = frame.sizes()[0] * frame.sizes()[1] * frame.sizes()[2];

  // We're gonna need these regardless of which device we're using
  uint8_t *compressedData = (uint8_t *) malloc(size);
  uint64_t compressedSize;

  bool cuda = frame.device() == torch::kCUDA; // Check device tensor
  // Not sure if this path gets used at all, need to investigate: TODO
  if(cuda){
        PRINTDBG("Using device memory\n");

    uint8_t *compressedDevData;
    cudaMalloc((void **)&compressedDevData, size);
    // Compress the frame
    compressedSize = NvPipe_Encode(encoder, data_ptr,
                                  info.width * info.depth,
                                  compressedDevData, size, info.width, info.height, false);

    // Can't just write to a file from device memory as far as i know
    cudaMemcpy(compressedData, compressedDevData, compressedSize, cudaMemcpyDeviceToHost);
    cudaFree(compressedDevData);
  } else {
    PRINTDBG("Using host memory\n");
    // Compress the frame
    compressedSize = NvPipe_Encode(encoder, data_ptr,
                                  info.width * info.depth,
                                  compressedData, size, info.width, info.height, false);
  }

  PRINTDBG("Writing %lu bytes to file\n", compressedSize);
  PRINTDBG("Comresison saves %ld bytes -> compressed data is %f %% smaller\n\n", size - compressedSize, (float)compressedSize/size * 100);

  // Make sure that the ROI is valid
  assert(ROI.sizes().size() == 2 && ROI.sizes()[1] == 4);

  if(ROI.scalar_type() != torch::kI32){
    ROI = ROI.toType(torch::kI32);
  }
  // Write the IV
  file.write((char *) &iv_size, sizeof(uint32_t));
  file.write(IV, iv_size);
  // Write the ROI(s)
  uint32_t roi_size = ROI.sizes()[0] * sizeof(uint32_t) * 4;
  file.write((char *) &roi_size, sizeof(uint32_t));
  file.write((char *) ROI.data_ptr<uint32_t>(), roi_size);
  // Write the actual video data
  file.write((char *) &compressedSize, sizeof(uint64_t));
  file.write((char *) compressedData, compressedSize);
}

// Cleanup
VideoEncoder::~VideoEncoder(){
  file.close();
  NvPipe_Destroy(encoder);
}

// TODO add docstrings through pybind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
py::class_<VideoEncoder>(m, "VideoEncoder")
  .def(py::init<std::string, uint32_t, uint32_t>())
  .def("write", &VideoEncoder::write);
}
