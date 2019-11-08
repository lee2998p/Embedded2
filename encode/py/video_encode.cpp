#include "video_encode.h"

#define DEBUG

#ifdef DEBUG
    #define DBG printf
#else
    #define DBG
#endif

void fatal(std::string msg){
  // Wanna point to the compare (usually right above)
  std::cerr << __LINE__-1 << ":" << __FILE__ << " " << msg << std::endl;
  exit(EXIT_FAILURE);
}

VideoEncoder::VideoEncoder(std::string base_name, uint32_t width, uint32_t height, bool ROI, bool ENC,
                            NvPipe_Codec codec, NvPipe_Compression compression, NvPipe_Format format,
                            uint32_t frame_rate, uint64_t bitrate){
  header.width = width;
  header.height = height;
  header.target_frame_rate = frame_rate;
  header.bitrate = bitrate;
  stride = width;

  if(codec == NVPIPE_HEVC)
    header.options |= O_CODEC_HEVC;
  if(compression == NVPIPE_LOSSLESS)
    header.options |= O_COMP_LOSSLESS;
  if(format == NVPIPE_RGBA32){
    header.options = O_FMT_RGBA;
    stride *= 4;
  }
  if(ROI)
    header.options  |= O_ENABLE_ROI;
  if(ENC)
    header.options |= O_ENABLE_ENCRYPT;


  DBG("[%s] Header options are: %d\n", __FILE__, (int) header.options);

  encoder = NvPipe_CreateEncoder(format, codec, compression, bitrate, frame_rate, header.width, header.height);

  if(!encoder){
    std::cerr << "[!!] Failed to create encoder, please ensure hardware is available" << std::endl;
    exit(EXIT_FAILURE);
  }

  DBG("[%s] Created encoder\n", __FILE__);

  filename = base_name;
  if(codec == NVPIPE_HEVC)
    filename += ".hevc";
  else
    filename += ".h264";

  file = std::ofstream(filename, std::ios::binary | std::ios::out | std::ios::trunc);
  DBG("[%s] Opened file %s\n", __FILE__, filename.c_str());
  file.write((char *)&header, sizeof(file_header));
  DBG("[%s] Wrote header to %s\n", __FILE__, filename.c_str());

}

bool VideoEncoder::check_enable(uint8_t opt){
  uint8_t mask;
  if(opt == O_FMT_RGBA){
    mask = 0b11111011;
  }
  if(opt == O_ENABLE_ROI){
    mask = 0b11110111;
  } if(opt == O_ENABLE_ENCRYPT){
    mask = 0b11101111;
  }
  return (header.options ^ mask) == opt;
}

void VideoEncoder::write(torch::Tensor frame){
  write(frame, torch::zeros({0, 4}));
}

void VideoEncoder::write(torch::Tensor frame, torch::Tensor ROI){
  write(frame, ROI, nullptr, 0);
}

void VideoEncoder::write(torch::Tensor frame, torch::Tensor ROI, char *IV, uint32_t iv_size){
  if(frame.scalar_type() != torch::kU8)
    fatal("Invalid type for frame");
  // Check if RGBA, make sure that the A channel exists
  if(frame.size(2) == 3 && check_enable(O_FMT_RGBA)){
    DBG("[%s] Adding alpha channel to frame\n", __FILE__);
    // Add A channel as fully in use if not given already
    // Create a tensor to match frame, and add to the last dimension
    torch::Tensor alpha = torch::full({frame.sizes()[0], frame.sizes()[1], 1}, 255, frame.options());
    frame = torch::cat({frame, alpha}, 2);
  }
  uint8_t *frame_ptr = frame.data_ptr<uint8_t>();
  uint64_t size = frame.size(0) * frame.size(1) * frame.size(2);

  uint8_t *compressedData = (uint8_t *) malloc(size);
  uint64_t compressedSize;

  // If the frame is held on device, must copy to host memory after encoding
  if(frame.device() == torch::kCUDA){
    DBG("[%s] Encoding from device memory\n", __FILE__);
    uint8_t *compressedDevData;
    cudaMalloc((void **)&compressedDevData, size);
    compressedSize = NvPipe_Encode(encoder, frame_ptr, stride, compressedDevData, size, header.width, header.height, false);
    // Copy the compressed data to host memory and free the device data
    cudaMemcpy(compressedData, compressedDevData, compressedSize, cudaMemcpyDeviceToHost);
    cudaFree(compressedDevData);
  } else {
    //DBG("[%s] Encoding from host memeory\n", __FILE__);
    compressedSize = NvPipe_Encode(encoder, frame_ptr, stride, compressedData, size, header.width, header.height, false);
  }
  // Handle ROIs
  if(check_enable(O_ENABLE_ROI)){
    if(ROI.sizes().size() != 0 || (ROI.sizes().size() != 2 && ROI.size(1)==4))
      fatal("Shape of ROI invalid, must be 0 or Nx4");
    if(ROI.scalar_type() != torch::kI32)
      fatal("ROI scalar type invalid, must be kI32");

    uint32_t roi_size = ROI.size(0) * 4 * sizeof(int32_t);
    uint8_t *roi_data = ROI.data_ptr<uint8_t>();
    file.write((char*) &roi_size, sizeof(uint32_t));
    file.write((char *) roi_data, roi_size);
  }
  // Handle encryption
  if(check_enable(O_ENABLE_ENCRYPT)){
    if(iv_size != 0 && !IV)
      fatal("IV is null and iv_size is non-zero");

    file.write((char *)&iv_size, sizeof(uint32_t));
    file.write(IV, iv_size);
  }

  file.write((char *)&compressedSize, sizeof(uint64_t));
  file.write((char *)compressedData, compressedSize);

  free(compressedData);
  frame_count+=1;
}



// Cleanup
VideoEncoder::~VideoEncoder(){
  file.close();
  NvPipe_Destroy(encoder);
  DBG("[%s] Closed file and destroyed encoder\n", __FILE__);
}

int32_t VideoEncoder::getFramesWritten(){
  return frame_count;
}


VideoDecoder::VideoDecoder(std::string filename){
  file = std::ifstream(filename, std::ios::binary | std::ios::in);
  file.read((char *) &header, sizeof(file_header));
  NvPipe_Codec codec = check_enable(O_CODEC_HEVC) ? NVPIPE_HEVC : NVPIPE_H264;
  NvPipe_Format fmt = check_enable(O_FMT_RGBA) ? NVPIPE_RGBA32 : NVPIPE_UINT8;
  decoder = NvPipe_CreateDecoder(fmt, codec, header.width, header.height);
}

// TODO handle ROI and IV data return
torch::Tensor VideoDecoder::read_frame(){
  char *IV;
  uint32_t iv_size;
  uint32_t roi_count;
  uint32_t *roi_data;
  if(check_enable(O_ENABLE_ROI)){
    file.read((char *)&roi_count, sizeof(uint32_t));
    file.read((char *)roi_data, sizeof(uint32_t) * roi_count * 4);
    DBG("[%s] Read ROI\n", __FILE__);
  }

  if(check_enable(O_ENABLE_ENCRYPT)){
    file.read((char *)&iv_size, sizeof(uint32_t));
    file.read((char *)IV, iv_size);
    DBG("[%s] Read IV\n", __FILE__);
  }
  uint64_t compressedSize;
  uint8_t *compressedData;

  file.read((char *) &compressedSize, sizeof(uint64_t));
  DBG("[%s] Read compressed data size\n", __FILE__);
  file.read((char *) compressedData, compressedSize);
  DBG("[%s] Read compressed data\n", __FILE__);
  uint64_t frame_size = header.height * header.width * 4;
  uint8_t *uncompressedData = (uint8_t *) malloc(frame_size);
  DBG("[%s] Decoded frame\n", __FILE__);
  NvPipe_Decode(decoder, compressedData, compressedSize, uncompressedData, header.width, header.height);

  torch::TensorOptions t_opts = torch::TensorOptions()
    .dtype(torch::kU8)
    .device(torch::kCPU, 1)
    .requires_grad(false);
  // Not so sure about this bit
  DBG("[%s] Converting frame array to tensor\n", __FILE__);
  torch::Tensor frame = torch::from_blob(uncompressedData, {header.width, header.height, 3}, {1,1,1}, free, t_opts);
  frame_count += 1;
  return frame;

}

VideoDecoder::~VideoDecoder(){
  file.close();
  NvPipe_Destroy(decoder);
  DBG("[%s] Closed file and destroyed decoder\n", __FILE__);
}

bool VideoDecoder::check_enable(uint8_t opt){
  uint8_t mask;
  if(opt == O_FMT_RGBA){
    mask = 0b11111011;
  }
  if(opt == O_ENABLE_ROI){
    mask = 0b11110111;
  } if(opt == O_ENABLE_ENCRYPT){
    mask = 0b11101111;
  }
  return (header.options ^ mask) == opt;
}

// TODO add docstrings through pybind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  py::class_<VideoEncoder>(m, "VideoEncoder")
    .def(py::init<std::string, uint32_t, uint32_t, bool, bool>(), py::arg("filename"), py::arg("width"), py::arg("height"), py::arg("ROI")=true, py::arg("ENCODE")=true)
    .def("write", (void (VideoEncoder::*)(torch::Tensor)) &VideoEncoder::write, py::arg("frame"))
    .def("write", (void (VideoEncoder::*)(torch::Tensor, torch::Tensor))
        &VideoEncoder::write, py::arg("frame"), py::arg("ROI"))
    .def("write", (void (VideoEncoder::*)(torch::Tensor, torch::Tensor, char *, uint32_t))
        &VideoEncoder::write, py::arg("frame"), py::arg("ROI"), py::arg("IV"), py::arg("iv_size"))
    .def("get_frame_count", &VideoEncoder::getFramesWritten);

//  py::class_<VideoDecoder>(m, "VideoDecoder")
//    .def(py::init<std::string>(), py::arg("filename"))
//    .def("read", &VideoDecoder::read_frame);

}
