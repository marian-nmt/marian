#include <chrono>

#include "god.h"
#include "decoder/encoder_decoder.h"
#include "decoder/sentence.h"
#include "dl4mt.h"

int main(int argc, char* argv[]) {
  God::Init(argc, argv);

  auto scorers = God::GetScorers(0);
  EncoderDecoder& encdec = *std::static_pointer_cast<EncoderDecoder>(scorers[0]);
  Encoder& encoder = encdec.GetEncoder();
  Decoder& decoder = encdec.GetDecoder();

  Sentence s(0, "das ist ein kleiner Test .");
  for(auto& w : s.GetWords())
    std::cerr << w << std::endl;

  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  for(size_t i = 0; i < 1; ++i) {
    mblas::Matrix context;
    encoder.GetContext(s.GetWords(), context);
    mblas::Debug(context);
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::duration<double> fp_s = end - start;
  LOG(progress) << fp_s.count() << "s";

  return 0;
}
