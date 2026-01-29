#include <anna/anna.hpp>
#include <iostream>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sndfile.h>

#define buffer_size 128
#define process_size 64
// #define bench_nframes (48000*100)
#define bench_nframes (64 * 4096)

#define nchannels 16
int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    std::cout << "Usage: " << argv[0] << " nam input output\n";
    return 1;
  }
  

	anna::sequence::input_output::model<
      anna::conv1d_double_buffer<float, buffer_size, 1, 1, nchannels, true, 1>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 1>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 2>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 4>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 8>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 16>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 32>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 64>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 128>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 256>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 512>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 1>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 2>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 4>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 8>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 16>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 32>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 64>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 128>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 256>,
      anna::conv1d_double_buffer<float, buffer_size, 3, nchannels, nchannels, true, 512>,
      anna::conv1d_double_buffer<float, buffer_size, 1, nchannels, 1, true, 512>
    > model;

	/*
	anna::sequence::input_output::model<
		  anna::conv1d<float, buffer_size, 1, 1, 2, true, 1>
		, 
			anna::conv1d<float, buffer_size, 1, 2, 1, true, 1>
	> model;
	*/

  std::ifstream f(argv[1]);
  nlohmann::json data = nlohmann::json::parse(f);

  // model.set_parameters(data["weights"].get<std::vector<float>>());

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

  for (size_t run = 0; run < 1; ++run)
  {
  for (long idx = 0; idx < sf_info.frames/process_size; ++idx) {
    input.template leftCols(process_size) = Eigen::Map<Eigen::Matrix<float, 1, process_size>>(input_file.data() + idx * process_size);
    Eigen::Map<Eigen::Matrix<float, 1, process_size>> output(output_file.data() + idx * process_size);
    /* output = */ model.process(input, process_size);
    // output = model.get_output().template leftCols(process_size);
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
