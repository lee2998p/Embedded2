#include <torch/extension.h>
#include <NvPipe.h>
#include <vector>
#include <string>
#include <cuda_runtime_api.h>

// TODO write unit tests
typedef struct {
  uint32_t width;
  uint32_t height;
  uint8_t depth;
} frame_info;


class VideoEncoder{
public:
  VideoEncoder(std::string base_name, uint32_t width, uint32_t height,
      NvPipe_Codec codec = NVPIPE_HEVC, NvPipe_Compression compression = NVPIPE_LOSSLESS,
      NvPipe_Format format = NVPIPE_RGBA32,
      uint32_t frame_rate = 30, uint64_t bitrate = 32 *1000 *1000);

  ~VideoEncoder();

  void write(torch::Tensor f);

private:
  std::string extension, filename;
  frame_info info;
  NvPipe *encoder;
  std::ofstream file;
  uint64_t dataPitch;
};
