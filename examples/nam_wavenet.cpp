#include "nam_wavenet.hpp"
#include <iostream>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sndfile.h>

#define process_size 64
#define max_buffer_size 64

int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    std::cout << "Usage: " << argv[0] << " nam input output\n";
    return 1;
  }
  
  auto *model = new anna::examples::nam_wavenet<float, 1, 1, 16, 3, 8, 3, max_buffer_size>();

  std::ifstream f(argv[1]);
  nlohmann::json data = nlohmann::json::parse(f);

  std::vector<float> params = data["weights"].get<std::vector<float>>();
  std::cout << "# of weights: " <<  params.size() << "\n";

  size_t idx = 0;
  model->set_parameters(params, idx);

  Eigen::Matrix<float, 1, 64> zero_input = Eigen::Matrix<float, 1, 64>::Zero();
  Eigen::Matrix<float, 1, 64> zero_output;

  for (int index = 0; index < 128; ++index)
    model->process(zero_input, zero_output, 64);

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

  Eigen::Matrix<float, 1, process_size> input = Eigen::Matrix<float, 1, process_size>::Zero();

  std::cout << "Processing: " << process_size * (sf_info.frames/process_size) << " samples\n";

  auto tick = std::chrono::high_resolution_clock::now();

  for (long idx = 0; idx < sf_info.frames/process_size; ++idx) {
    input.template leftCols(process_size) = Eigen::Map<Eigen::Matrix<float, 1, process_size>>(input_file.data() + idx * process_size);
    Eigen::Map<Eigen::Matrix<float, 1, process_size>> output(output_file.data() + idx * process_size);
    model->process(input, output, process_size);
  }
  
  auto tock = std::chrono::high_resolution_clock::now();

  float duration = std::chrono::duration_cast<std::chrono::duration<float>>(tock - tick).count();

  std::cout << "Elapsed: " << duration << " s\n";
  std::cout << "Elapsed per second of audio: " << sf_info.samplerate * duration / ((double)process_size * (sf_info.frames/process_size)) << "\n";

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
