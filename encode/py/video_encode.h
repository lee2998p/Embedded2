#include <torch/extension.h>
#include <NvPipe.h>
#include <vector>
#include <string>
#include <cuda_runtime_api.h>


#define O_CODEC_HEVC      0b00000001
#define O_COMP_LOSSLESS   0b00000010
#define O_FMT_RGBA        0b00000100
#define O_ENABLE_ROI      0b00001000
#define O_ENABLE_ENCRYPT  0b00010000

typedef struct {
  uint32_t width;
  uint32_t height;
  uint8_t options;
  uint32_t target_frame_rate;
  uint64_t bitrate;
} file_header;


class VideoEncoder{
public:
  VideoEncoder(std::string base_name, uint32_t width, uint32_t height, bool ROI = false, bool ENC = false,
      NvPipe_Codec codec = NVPIPE_HEVC, NvPipe_Compression compression = NVPIPE_LOSSLESS,
      NvPipe_Format format = NVPIPE_RGBA32,
      uint32_t frame_rate = 30, uint64_t bitrate = 32 *1000 *1000);

  ~VideoEncoder();

  void write(torch::Tensor frame);

  void write(torch::Tensor frame, torch::Tensor ROI);

  /**
   * Writes an image, given as a torch tensor to a file
   * @param frame the frame to write as a [NxMx3] or [NxMx4] torch tensor, element type is uint8
   * @param ROI optional tensor containing regions of interest in the frame, a [Nx4] torch tensor, element type is int32
   * @param IV a buffer of characters or null
   * @param iv_size non-zero size of IV buffer if used
   */
  void write(torch::Tensor frame, torch::Tensor ROI, char *IV, uint32_t iv_size);

  int32_t getFramesWritten();

private:
  bool check_enable(uint8_t opt);
  uint32_t frame_count = 0;
  std::string filename;
  NvPipe *encoder;
  std::ofstream file;
  file_header header;
  uint32_t stride;
};

class VideoDecoder{
public:
  VideoDecoder(std::string filename);
  ~VideoDecoder();
  torch::Tensor read_frame();
private:
  bool check_enable(uint8_t opt);
  uint32_t frame_count = 0;
  NvPipe *decoder;
  std::ifstream file;
  file_header header;
  uint32_t stride;
};
