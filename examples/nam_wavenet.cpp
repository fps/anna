#include <anna/nam.hpp>
#include <iostream>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sndfile.h>

#define buffer_size 128
#define process_size 64
// #define bench_nframes (48000*100)
#define bench_nframes (64 * 4096)

int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    std::cout << "Usage: " << argv[0] << " nam input output\n";
    return 1;
  }
  
  anna::nam::wavenet<
    float, buffer_size,
    1, 1,
    anna::nam::wavenet_block<
      float, buffer_size, 1, 1, 16, 8, false,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 1, false, true>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 2>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 4>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 8>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 16>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 32>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 64>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 128>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 256>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 512>
      >,
    anna::nam::wavenet_block<
      float, buffer_size, 1, 16, 8, 1, true,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 1>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 2>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 4>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 8>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 16>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 32>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 64>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 128>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 256>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 512, true>
      >
    > model;

  std::ifstream f(argv[1]);
  nlohmann::json data = nlohmann::json::parse(f);

  model.set_parameters(data["weights"].get<std::vector<float>>());

  SF_INFO sf_info = { 0 };
  SNDFILE *sndfile = sf_open(argv[2], SFM_READ, &sf_info);
  if (nullptr == sndfile)
  {
    std::cout << "Failed to open: " << argv[2] << "\n";
    return 1;
  }

  if (sf_info.channels != 1)
  {
    std::cout << "Only mono files are supported\n";
    return 1;
  }

  std::vector<float> input_file(sf_info.frames);
  sf_count_t count = sf_readf_float(sndfile, input_file.data(), sf_info.frames);
  if (count != sf_info.frames)
  {
    std::cout << "Failed to read input file: " << argv[2] << ": Wrong number of frames.\n";
    return 1;
  }

  sf_close(sndfile);

  std::vector<float> output_file(sf_info.frames, 0);

  Eigen::Matrix<float, 1, buffer_size> input = Eigen::Matrix<float, 1, buffer_size>::Zero();

  std::cout << "Processing: " << process_size * (sf_info.frames/process_size) << " samples\n";

  auto tick = std::chrono::high_resolution_clock::now();

  for (size_t run = 0; run < 10; ++run)
  {
  for (long idx = 0; idx < sf_info.frames/process_size; ++idx) {
    input.template leftCols(process_size) = Eigen::Map<Eigen::Matrix<float, 1, process_size>>(input_file.data() + idx * process_size);
    model.process(input, process_size);
    Eigen::Map<Eigen::Matrix<float, 1, process_size>> output(output_file.data() + idx * process_size);
    output = model.get_output().template leftCols(process_size);
  }
  }
  
  /*
  for (long idx = 0; idx < sf_info.frames/process_size; ++idx) {
    model.process(input, process_size);
    // gutput = model.get_output().template leftCols(process_size);
  }
  */

  auto tock = std::chrono::high_resolution_clock::now();

  std::cout << (tock - tick).count() << "\n";

  SF_INFO sf_info2;
  sf_info2.channels = 1;
  sf_info2.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
  sf_info2.samplerate = sf_info.samplerate;
  
  SNDFILE *sndfile2 = sf_open(argv[3], SFM_WRITE, &sf_info2);
  if (nullptr == sndfile2)
  {
    std::cout << "Failed to open: " << argv[3] << "\n";
    return 1;
  }

  sf_writef_float(sndfile2, output_file.data(), sf_info.frames);
  // sf_writef_float(sndfile2, input_file.data(), sf_info.frames);
  sf_close(sndfile2);
}
