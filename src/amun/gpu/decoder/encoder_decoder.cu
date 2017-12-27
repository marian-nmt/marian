// -*- mode: c++; tab-width: 2; indent-tabs-mode: nil -*-
#include <iostream>

#include "common/god.h"
#include "common/sentences.h"
#include "common/search.h"
#include "common/histories.h"
#include "common/hypothesis.h"

#include "encoder_decoder.h"
#include "enc_out_gpu.h"
#include "gpu/mblas/matrix_functions.h"
#include "gpu/dl4mt/dl4mt.h"
#include "gpu/decoder/encoder_decoder_state.h"
#include "gpu/decoder/best_hyps.h"
#include "gpu/dl4mt/cellstate.h"


using namespace std;

namespace amunmt {
namespace GPU {

std::unordered_map<std::string, boost::timer::cpu_timer> timers;

EncoderDecoder::EncoderDecoder(
		const God &god,
		const std::string& name,
        const YAML::Node& config,
        size_t tab,
        const Weights& model,
        const Search &search)
  : Scorer(god, name, config, tab, search),
    model_(model),
    encoder_(new Encoder(model_, config)),
    decoder_(new Decoder(god, model_, config)),
    encDecBuffer_(god.Get<size_t>("encoder-buffer-size"))
{
  BEGIN_TIMER("EncoderDecoder");

  std::thread *thread = new std::thread( [&]{ DecodeAsync(); });
  decThread_.reset(thread);
}

EncoderDecoder::~EncoderDecoder()
{
  //cerr << "~EncoderDecoder1" << endl;
  decThread_->join();
  PAUSE_TIMER("EncoderDecoder");

  if (timers.size()) {
    boost::timer::nanosecond_type encDecWall = timers["EncoderDecoder"].elapsed().wall;

    cerr << "timers:" << endl;
    for (auto iter = timers.begin(); iter != timers.end(); ++iter) {
      const boost::timer::cpu_timer &timer = iter->second;
      boost::timer::cpu_times t = timer.elapsed();
      boost::timer::nanosecond_type wallTime = t.wall;

      int percent = (float) wallTime / (float) encDecWall * 100.0f;

      cerr << iter->first << " ";

      for (int i = 0; i < ((int)35 - (int)iter->first.size()); ++i) {
        cerr << " ";
      }

      cerr << timer.format(2, "%w") << " (" << percent << "%)" << endl;
    }
  }

  //cerr << "~EncoderDecoder2" << endl;
}

void EncoderDecoder::Encode(const SentencesPtr &source) {
  BEGIN_TIMER("Encode");

  EncOutPtr encOut(new EncOutGPU(source));

  if (source->size()) {
    encoder_->Encode(encOut, tab_);
  }

  encDecBuffer_.Add(encOut);


  PAUSE_TIMER("Encode");
}

State* EncoderDecoder::NewState() const {
  return new EDState();
}

void EncoderDecoder::GetAttention(mblas::Matrix& Attention) {
  decoder_->GetAttention(Attention);
}

BaseMatrix& EncoderDecoder::GetProbs() {
  return decoder_->GetProbs();
}

void *EncoderDecoder::GetNBest()
{
  return &decoder_->GetNBest();
}

const BaseMatrix *EncoderDecoder::GetBias() const
{
  return decoder_->GetBias();
}

mblas::Matrix& EncoderDecoder::GetAttention() {
  return decoder_->GetAttention();
}

size_t EncoderDecoder::GetVocabSize() const {
  return decoder_->GetVocabSize();
}

void EncoderDecoder::Filter(const std::vector<uint>& filterIds) {
  decoder_->Filter(filterIds);
}

/////////////////////////////////////////////////////////////////////////////////////
// const-batch2
void EncoderDecoder::DecodeAsync()
{
  //cerr << "BeginSentenceState encOut->sourceContext_=" << encOut->sourceContext_.Debug(0) << endl;
  try {
    DecodeAsyncInternal();
  }
  catch(thrust::system_error &e)
  {
    std::cerr << "DecodeAsync: CUDA error during some_function: " << e.what() << std::endl;
    abort();
  }
  catch(std::bad_alloc &e)
  {
    std::cerr << "DecodeAsync: Bad memory allocation during some_function: " << e.what() << std::endl;
    abort();
  }
  catch(std::runtime_error &e)
  {
    std::cerr << "DecodeAsync: Runtime error during some_function: " << e.what() << std::endl;
    abort();
  }
  catch(...)
  {
    std::cerr << "DecodeAsync: Some other kind of error during some_function" << std::endl;
    abort();
  }
}

void EncoderDecoder::DecodeAsyncInternal()
{
  uint maxBeamSize = god_.Get<uint>("beam-size");
  uint miniBatch = god_.Get<uint>("mini-batch");

  Histories histories(search_.NormalizeScore());

  mblas::Vector<uint> sentenceLengths;
  mblas::Matrix sourceContext, SCU;
  StatePtr state, nextState;
  Hypotheses prevHyps;

  state.reset(NewState());
  nextState.reset(NewState());

  InitBatch(histories, sentenceLengths, sourceContext, SCU, *state, prevHyps);

  unsigned step = 0;
  while (histories.GetNumActive()) {
    boost::timer::cpu_timer timerStep;
    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "DecodeAsyncInternal1" << endl;
    //std::cerr << "histories1=" << histories.Debug(1) << std::endl;

    const EDState& edstate = state->get<EDState>();
    EDState& ednextState = nextState->get<EDState>();

    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "DecodeAsyncInternal2" << endl;
    //std::cerr << "histories2=" << histories.Debug(1) << std::endl;
    decoder_->Decode(ednextState.GetStates(),
                     edstate.GetStates(),
                     edstate.GetEmbeddings(),
                     histories,
                     god_.UseFusedSoftmax(),
                     sourceContext,
                     SCU,
                     sentenceLengths);
    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "DecodeAsyncInternal3" << endl;
    //std::cerr << "histories3=" << histories.Debug(1) << std::endl;

    histories.SetNewBeamSize(maxBeamSize);
    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "DecodeAsyncInternal4" << endl;
    //std::cerr << "histories4=" << histories.Debug(1) << std::endl;

    unsigned numPrevHyps = prevHyps.size();
    size_t survivors = CalcBeam(search_.GetBestHyps(), histories, prevHyps, *state, *nextState, search_.GetFilterIndices());
    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "DecodeAsyncInternal5" << endl;
    //std::cerr << "histories5=" << histories.Debug(1) << std::endl;

    if (survivors == 0) {
    //if (survivors < 10) {
      //AssembleBeamState(histories, *nextState, prevHyps, *state);

      //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      //cerr << "DecodeAsyncInternal6" << endl;
      //std::cerr << "histories6=" << histories.Debug(1) << std::endl;

      InitBatch(histories, sentenceLengths, sourceContext, SCU, *state, prevHyps);
      //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      //cerr << "DecodeAsyncInternal7" << endl;
      //std::cerr << "histories7=" << histories.Debug(1) << std::endl;
    }
    else {
      AssembleBeamState(*nextState, prevHyps, *state);
    }

    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "DecodeAsyncInternal8" << endl;
    //std::cerr << "histories8=" << histories.Debug(1) << std::endl;

    LOG(progress)->info("  Step {} took {} sentences {} prevHypos {} survivors {}", step++, timerStep.format(5, "%w"), histories.GetNumActive(), numPrevHyps, survivors);
  }
}

void EncoderDecoder::InitBatch(Histories &histories,
                                mblas::Vector<uint> &sentenceLengths,
                                mblas::Matrix &sourceContext,
                                mblas::Matrix &SCU,
                                State &state,
                                Hypotheses &prevHyps)
{
  ///*
  uint miniBatch = god_.Get<uint>("mini-batch");

  std::vector<BufferOutput> newSentences;
  encDecBuffer_.Get(miniBatch, newSentences);

  //vector<unsigned> batchIds = AddToBatch(newSentences, sentences, histories, sentenceLengths, sourceContext);

  if (newSentences.size() == 0) {
    return;
  }

  const EncOutPtr &encOut = newSentences.front().GetEncOut();
  assert(encOut);
  //*/
  /*
  EncOutPtr encOut = encDecBuffer_.Get();
  assert(encOut);

  sentences = encOut->GetSentences();
  if (sentences.size() == 0) {
      return false;
  }
  */

  //sentenceLengths = encOut->Get<EncOutGPU>().GetSentenceLengths();
  sentenceLengths.newSize(encOut->Get<EncOutGPU>().GetSentenceLengths().size());
  mblas::copy(encOut->Get<EncOutGPU>().GetSentenceLengths().data(),
              sentenceLengths.size(),
              sentenceLengths.data(),
              cudaMemcpyDeviceToHost);

  //sourceContext = encOut->Get<EncOutGPU>().GetSourceContext();
  const mblas::Matrix &origSourceContext = encOut->Get<EncOutGPU>().GetSourceContext();
  sourceContext.NewSize(origSourceContext.dim(0), origSourceContext.dim(1), origSourceContext.dim(2), origSourceContext.dim(3));
  mblas::CopyMatrix(sourceContext, origSourceContext);

  histories.Init(newSentences);

  BeginSentenceState(histories.GetNumActive(), sourceContext, sentenceLengths, state, SCU);

  prevHyps = histories.GetFirstHyps();

  return;
}

//////////////////////////////////////////////////////////////////////
//helper fn
void FindNextEmptyIndex(size_t &batchInd,
                        Histories &histories)
{
  while(batchInd < histories.size()) {
    const HistoriesElementPtr &ele = histories.Get(batchInd);
    if (ele == nullptr) {
      return;
    }
    ++batchInd;
  }

  assert(false);
  batchInd = 9999999;
}

////////////////////////////////////////////////////////////////////////

void EncoderDecoder::FetchBatch(Histories &histories,
                                mblas::Vector<uint> &sentenceLengths,
                                mblas::Matrix &sourceContext,
                                mblas::Matrix &SCU,
                                State &state,
                                Hypotheses &prevHyps)
{
  boost::timer::cpu_timer timer;

  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));

  size_t numSentToGet = god_.Get<uint>("mini-batch") - histories.GetNumActive();
  //cerr << "histories.GetNumActive()=" << histories.GetNumActive() << endl;

  std::vector<BufferOutput> newSentences;
  encDecBuffer_.Get(numSentToGet, newSentences);

  //vector<unsigned> batchIds = AddToBatch(newSentences, sentences, histories, sentenceLengths, sourceContext);

  if (newSentences.size() == 0) {
    return;
  }

  vector<uint> newBatchIds(newSentences.size());
  vector<uint> newSentenceLengths(newSentences.size());
  vector<uint> newSentenceOffsets(newSentences.size());

  // update existing batch
  size_t batchInd = 0;
  for (size_t i = 0; i < newSentences.size(); ++i) {
    const BufferOutput &eleSent = newSentences[i];
    const SentencePtr &sentence = eleSent.GetSentence();

    // work out offset in existing batch
    FindNextEmptyIndex(batchInd, histories);
    newBatchIds[i] = batchInd;

    // sentence lengths
    newSentenceLengths[i] = sentence->size();

    // offsets
    newSentenceOffsets[i] = eleSent.GetSentenceOffset();

    // histories
    histories.Set(batchInd, new HistoriesElement(sentence, histories.NormalizeScore()));

    ++batchInd;
  }

  size_t maxLength =  histories.MaxLength();
  //cerr << "maxLength=" << maxLength << endl;
  //cerr << "newBatchIds=" << Debug(newBatchIds, 2) << endl;

  // update gpu data
  mblas::Vector<uint> d_newBatchIds(newBatchIds);
  mblas::Vector<uint> d_newSentenceLengths(newSentenceLengths);

  UpdateSentenceLengths(d_newSentenceLengths, d_newBatchIds, sentenceLengths);

  // source context
  ResizeMatrix(sourceContext, {0, maxLength, 3, histories.GetNumActive()});

  AddNewData(sourceContext, newBatchIds, newSentences);

  BeginSentenceState(histories.GetNumActive(), sourceContext, sentenceLengths, state, SCU, newBatchIds, d_newBatchIds);

  prevHyps = histories.GetFirstHyps();

  LOG(progress)->info("Fetch took {} new {} histories {}", timer.format(5, "%w"), newSentences.size(), histories.GetNumActive());

  return;
}

void EncoderDecoder::BeginSentenceState(size_t batchSize,
                                        const mblas::Matrix &sourceContext,
                                        const mblas::Vector<uint> &sentenceLengths,
                                        State& state,
                                        mblas::Matrix& SCU) const
{
  //BEGIN_TIMER("BeginSentenceState");
  EDState& edState = state.get<EDState>();

  decoder_->EmptyState(edState.GetStates(), batchSize, sourceContext, sentenceLengths, SCU);

  decoder_->EmptyEmbedding(edState.GetEmbeddings(), batchSize);
  //PAUSE_TIMER("BeginSentenceState");
}

void EncoderDecoder::BeginSentenceState(size_t batchSize,
                                        const mblas::Matrix &sourceContext,
                                        const mblas::Vector<uint> &sentenceLengths,
                                        State& state,
                                        mblas::Matrix& SCU,
                                        const std::vector<uint> &newBatchIds,
                                        const mblas::Vector<uint> &d_newBatchIds) const
{
  //BEGIN_TIMER("BeginSentenceState");
  EDState& edState = state.get<EDState>();

  decoder_->EmptyState(edState.GetStates(), batchSize, sourceContext, sentenceLengths, SCU, newBatchIds, d_newBatchIds);

  decoder_->EmptyEmbedding(edState.GetEmbeddings(), batchSize, newBatchIds, d_newBatchIds);
  //PAUSE_TIMER("BeginSentenceState");
}

size_t EncoderDecoder::CalcBeam(BestHypsBase &bestHyps,
                      Histories& histories,
                      Hypotheses& prevHyps,
                      State& state,
                      State& nextState,
                      const Words &filterIndices)
{
  size_t batchSize = histories.size();
  HypothesesBatch beams(batchSize);
  bestHyps.CalcBeam(prevHyps, *this, filterIndices, beams, histories);

  //cerr << "beams=" << beams.size() << endl;
  assert(beams.size() == histories.size());
  assert(beams.size() == batchSize);

  Hypotheses survivors = histories.Add(god_, beams);

  if (survivors.size() == 0) {
    return 0;
  }

  //cerr << "survivors=" << survivors.size() << endl;
  prevHyps.swap(survivors);
  return prevHyps.size();
}

void EncoderDecoder::AssembleBeamState(const State& state,
                               const Hypotheses& beam,
                               State& nextState) const
{
  //BEGIN_TIMER("AssembleBeamState");
  std::vector<uint> beamWords;
  std::vector<uint> beamStateIds;
  for (const HypothesisPtr &h : beam) {
     beamWords.push_back(h->GetWord());
     beamStateIds.push_back(h->GetPrevStateIndex());
  }
  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  //cerr << "beamStateIds=" << Debug(beamStateIds, 2) << endl;

  const EDState& edState = state.get<EDState>();
  EDState& edNextState = nextState.get<EDState>();

  thread_local mblas::Vector<uint> indices;
  indices.newSize(beamStateIds.size());
  //mblas::Vector<uint> indices(beamStateIds.size());
  //cerr << "indices=" << indices.Debug(2) << endl;

  mblas::copy(beamStateIds.data(),
              beamStateIds.size(),
              indices.data(),
              cudaMemcpyHostToDevice);

  CellState& outstates = edNextState.GetStates();
  const CellState& instates = edState.GetStates();

  //cerr << "outstates.output=" << outstates.output->Debug(0) << endl;
  //cerr << "instates.output=" << instates.output->Debug(0) << endl;
  //cerr << "beamStateIds=" << Debug(beamStateIds, 2) << endl;
  //cerr << "indices=" << indices.Debug(2) << endl;

  mblas::Assemble(*(outstates.output), *(instates.output), indices);
  if (instates.cell->size() > 0) {
    mblas::Assemble(*(outstates.cell), *(instates.cell), indices);
  }
  //cerr << "edNextState.GetStates()=" << edNextState.GetStates().Debug(1) << endl;

  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  decoder_->Lookup(edNextState.GetEmbeddings(), beamWords);
  //cerr << "edNextState.GetEmbeddings()=" << edNextState.GetEmbeddings().Debug(1) << endl;
  //PAUSE_TIMER("AssembleBeamState");
}

void EncoderDecoder::AssembleBeamState(const Histories& histories,
                                        const State& state,
                                        const Hypotheses& beam,
                                        State& nextState) const
{
  //BEGIN_TIMER("AssembleBeamState");
  std::vector<uint> beamWords;
  std::vector<uint> beamStateIds;

  for (size_t i = 0; i < histories.size(); ++i) {
    const HistoriesElementPtr &ele = histories.Get(i);
    if (ele) {
      beamWords.push_back(0);
      beamStateIds.push_back(5);
    }
    else {
      for (const HypothesisPtr &h : beam) {
         beamWords.push_back(h->GetWord());
         beamStateIds.push_back(h->GetPrevStateIndex());
      }

    }
  }

  cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  cerr << "beamStateIds=" << Debug(beamStateIds, 2) << endl;

  const EDState& edState = state.get<EDState>();
  EDState& edNextState = nextState.get<EDState>();

  thread_local mblas::Vector<uint> indices;
  indices.newSize(beamStateIds.size());
  //mblas::Vector<uint> indices(beamStateIds.size());
  //cerr << "indices=" << indices.Debug(2) << endl;

  mblas::copy(beamStateIds.data(),
              beamStateIds.size(),
              indices.data(),
              cudaMemcpyHostToDevice);

  CellState& outstates = edNextState.GetStates();
  const CellState& instates = edState.GetStates();

  cerr << "outstates.output=" << outstates.output->Debug(0) << endl;
  cerr << "instates.output=" << instates.output->Debug(0) << endl;
  cerr << "beamStateIds=" << Debug(beamStateIds, 2) << endl;
  //cerr << "indices=" << indices.Debug(2) << endl;

  mblas::Assemble(*(outstates.output), *(instates.output), indices);
  if (instates.cell->size() > 0) {
    mblas::Assemble(*(outstates.cell), *(instates.cell), indices);
  }
  //cerr << "edNextState.GetStates()=" << edNextState.GetStates().Debug(1) << endl;

  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  decoder_->Lookup(edNextState.GetEmbeddings(), beamWords);
  //cerr << "edNextState.GetEmbeddings()=" << edNextState.GetEmbeddings().Debug(1) << endl;
  //PAUSE_TIMER("AssembleBeamState");
}

}
}

