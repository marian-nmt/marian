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

  state.reset(NewState());
  nextState.reset(NewState());

  //cerr << "prevHyps1=" << prevHyps.size() << endl;
  InitBatch(histories, sentenceLengths, sourceContext, SCU, *state);
  //cerr << "prevHyps2=" << prevHyps.size() << endl;

  unsigned step = 0;
  while (histories.NumActive()) {
    boost::timer::cpu_timer timerStep;
    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "DecodeAsyncInternal1" << endl;
    //std::cerr << "histories1=" << histories.Debug(1) << std::endl;

    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "DecodeAsyncInternal2" << endl;
    //std::cerr << "histories2=" << histories.Debug(1) << std::endl;
    //std::cerr << "state0=" << state->Debug(0) << std::endl;
    //std::cerr << "nextState0=" << nextState->get<EDState>().GetStates().output->Debug(0) << std::endl;
    //std::cerr << "embeddings=" << state->get<EDState>().GetEmbeddings().Debug(0) << std::endl;

    decoder_->Decode(nextState->get<EDState>().GetStates(),
                    state->get<EDState>().GetStates(),
                    state->get<EDState>().GetEmbeddings(),
                    histories,
                    god_.UseFusedSoftmax(),
                    sourceContext,
                    SCU,
                    sentenceLengths);
    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "DecodeAsyncInternal3" << endl;
    //std::cerr << "histories3=" << histories.Debug(1) << std::endl;
    //std::cerr << "state1=" << state->Debug(0) << std::endl;
    //std::cerr << "nextState1=" << nextState->get<EDState>().GetStates().output->Debug(0) << std::endl;

    histories.SetNewBeamSize(maxBeamSize);
    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "DecodeAsyncInternal4" << endl;
    //std::cerr << "histories4=" << histories.Debug(1) << std::endl;
    //std::cerr << "state2=" << state->Debug(0) << std::endl;
    //std::cerr << "nextState2=" << nextState->get<EDState>().GetStates().output->Debug(0) << std::endl;

    CalcBeam(search_.GetBestHyps(), histories, *state, *nextState, search_.GetFilterIndices());
    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "DecodeAsyncInternal5" << endl;
    //std::cerr << "histories5=" << histories.Debug(1) << std::endl;
    //std::cerr << "state3=" << state->Debug(0) << std::endl;
    //std::cerr << "nextState3=" << nextState->get<EDState>().GetStates().output->Debug(0) << std::endl;

    //if (histories.NumActive() == 0) {
    if ((histories.size() - histories.NumActive()) > 0) {
      //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      //cerr << "DecodeAsyncInternal6" << endl;
      //std::cerr << "histories6=" << histories.Debug(1) << std::endl;

      //InitBatch(histories, sentenceLengths, sourceContext, SCU, *state);
      FetchBatch(histories, sentenceLengths, sourceContext, SCU, *nextState, *state);
      //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      //cerr << "DecodeAsyncInternal7" << endl;
      //std::cerr << "histories7=" << histories.Debug(1) << std::endl;
      //std::cerr << "state4=" << state->Debug(0) << std::endl;
      //std::cerr << "nextState4=" << nextState->get<EDState>().GetStates().output->Debug(0) << std::endl;
    }
    else {
      AssembleBeamState(histories, *nextState, *state);
      //std::cerr << "state5=" << state->Debug(0) << std::endl;
      //std::cerr << "nextState5=" << nextState->get<EDState>().GetStates().output->Debug(0) << std::endl;
    }
    //std::cerr << "state6=" << state->Debug(0) << std::endl;
    //std::cerr << "nextState6=" << nextState->get<EDState>().GetStates().output->Debug(0) << std::endl;

    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "DecodeAsyncInternal8" << endl;
    //std::cerr << "histories8=" << histories.Debug(1) << std::endl;

    LOG(progress)->info("  Step {} took {} sentences {}", step++, timerStep.format(5, "%w"), histories.NumActive());
  }
}

void EncoderDecoder::InitBatch(Histories &histories,
                                mblas::Vector<uint> &sentenceLengths,
                                mblas::Matrix &sourceContext,
                                mblas::Matrix &SCU,
                                State &state)
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
  //cerr << "histories=" << histories.Debug() << endl;

  BeginSentenceState(histories, sourceContext, sentenceLengths, state, SCU);

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
                                State &nextState,
                                State &state)
{
  boost::timer::cpu_timer timer;

  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  cerr << "FetchBatch1" << endl;
  size_t numSentToGet = god_.Get<uint>("mini-batch") - histories.NumActive();
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "FetchBatch2" << endl;
  //cerr << "numSentToGet=" << numSentToGet << endl;
  //cerr << "histories orig=" << histories.Debug() << endl;

  std::vector<BufferOutput> newSentences;
  encDecBuffer_.Get(numSentToGet, newSentences);
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "FetchBatch3" << endl;
  //cerr << "newSentences=" << newSentences.size() << endl;

  vector<uint> newBatchIds(newSentences.size());
  vector<uint> newSentenceLengths(newSentences.size());
  vector<uint> newSentenceOffsets(newSentences.size());

  // update existing batch
  size_t batchInd = 0;
  for (size_t i = 0; i < newSentences.size(); ++i) {
    //cerr << "FetchBatch4" << endl;
    const BufferOutput &eleSent = newSentences[i];
    const SentencePtr &sentence = eleSent.GetSentence();

    // work out offset in existing batch
    FindNextEmptyIndex(batchInd, histories);
    //cerr << "batchInd=" << batchInd << endl;

    assert(i < newBatchIds.size());
    newBatchIds[i] = batchInd;

    // sentence lengths
    assert(i < newSentenceLengths.size());
    newSentenceLengths[i] = sentence->size();

    // offsets
    assert(i < newSentenceOffsets.size());
    newSentenceOffsets[i] = eleSent.GetSentenceOffset();

    // histories
    histories.Set(batchInd, new HistoriesElement(sentence, histories.NormalizeScore()));

    ++batchInd;
  }
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "FetchBatch5" << endl;

  if (histories.NumActive() == 0) {
    return;
  }

  size_t maxLength =  histories.MaxLength();
  //cerr << "maxLength=" << maxLength << endl;
  //cerr << "newBatchIds=" << Debug(newBatchIds, 2) << endl;

  // update gpu data
  mblas::Vector<uint> d_newBatchIds(newBatchIds);
  mblas::Vector<uint> d_newSentenceLengths(newSentenceLengths);
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "FetchBatch6" << endl;
  //cerr << "histories=" << histories.Debug() << endl;
  //cerr << "1state=" << state.Debug(0) << endl;

  AssembleBeamState(newBatchIds, d_newBatchIds, histories, nextState, state);
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "FetchBatch7" << endl;
  //cerr << "2state=" << state.Debug(0) << endl;

  UpdateSentenceLengths(d_newSentenceLengths, d_newBatchIds, sentenceLengths);
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "FetchBatch8" << endl;

  // source context
  //cerr << "1sourceContext=" << sourceContext.Debug() << endl;
  ResizeMatrix(sourceContext, {0, maxLength});
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "FetchBatch9" << endl;
  //cerr << "2sourceContext=" << sourceContext.Debug() << endl;

  AddNewData(sourceContext, newBatchIds, newSentences);
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "FetchBatch10" << endl;

  BeginSentenceState(histories, sourceContext, sentenceLengths, state, SCU, newBatchIds, d_newBatchIds);
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "FetchBatch11" << endl;
  //cerr << "histories new=" << histories.Debug() << endl;

  LOG(progress)->info("Fetch took {} new {} histories {}", timer.format(5, "%w"), newSentences.size(), histories.NumActive());
  cerr << endl;
}

void EncoderDecoder::BeginSentenceState(const Histories& histories,
                                        const mblas::Matrix &sourceContext,
                                        const mblas::Vector<uint> &sentenceLengths,
                                        State& state,
                                        mblas::Matrix& SCU) const
{
  //BEGIN_TIMER("BeginSentenceState");
  EDState& edState = state.get<EDState>();
  size_t batchSize = histories.NumActive();

  decoder_->EmptyState(edState.GetStates(), histories, sourceContext, sentenceLengths, SCU);

  decoder_->EmptyEmbedding(edState.GetEmbeddings(), batchSize);
  //PAUSE_TIMER("BeginSentenceState");
}

void EncoderDecoder::BeginSentenceState(const Histories& histories,
                                        const mblas::Matrix &sourceContext,
                                        const mblas::Vector<uint> &sentenceLengths,
                                        State& state,
                                        mblas::Matrix& SCU,
                                        const std::vector<uint> &newBatchIds,
                                        const mblas::Vector<uint> &d_newBatchIds) const
{
  //BEGIN_TIMER("BeginSentenceState");
  EDState& edState = state.get<EDState>();
  size_t batchSize = histories.NumActive();

  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "BeginSentenceState1" << endl;

  decoder_->EmptyState(edState.GetStates(), histories, sourceContext, sentenceLengths, SCU, newBatchIds, d_newBatchIds);
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "BeginSentenceState2" << endl;

  decoder_->EmptyEmbedding(edState.GetEmbeddings(), histories.GetTotalBeamSize(), newBatchIds, d_newBatchIds);
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "BeginSentenceState3" << endl;
  //PAUSE_TIMER("BeginSentenceState");
}

void EncoderDecoder::CalcBeam(BestHypsBase &bestHyps,
                      Histories& histories,
                      State& state,
                      State& nextState,
                      const Words &filterIndices)
{
  //cerr << "CalcBeam1" << endl;
  //cerr << "1histories=" << histories.Debug() << endl;
  Hypotheses prevHypos = histories.GetSurvivors();
  //cerr << "CalcBeam2" << endl;
  //cerr << "prevHypos=" << prevHypos.size() << endl;

  histories.StartCalcBeam();
  //cerr << "CalcBeam3" << endl;
  bestHyps.CalcBeam(prevHypos, *this, filterIndices, histories);
  //cerr << "CalcBeam4" << endl;
  histories.Add(god_);
  //cerr << "CalcBeam5" << endl;
}

void EncoderDecoder::AssembleBeamState(const State& state,
                               const Hypotheses& beam,
                               State& nextState) const
{
  assert(false);
}

void EncoderDecoder::AssembleBeamState(const Histories& histories,
                                        const State& inState,
                                        State& outState) const
{
  //BEGIN_TIMER("AssembleBeamState");
  std::vector<uint> beamWords;
  std::vector<uint> beamStateIds;
  for (size_t i = 0; i < histories.size(); ++i) {
    const HistoriesElementPtr &ele = histories.Get(i);
    if (ele) {
      const Hypotheses &beam = ele->GetHypotheses();
      for (const HypothesisPtr &h : beam) {
         beamWords.push_back(h->GetWord());
         beamStateIds.push_back(h->GetPrevStateIndex());
      }
    }
  }
  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  //cerr << "beamStateIds=" << Debug(beamStateIds, 2) << endl;

  const EDState& edInState = inState.get<EDState>();
  EDState& edOutState = outState.get<EDState>();

  thread_local mblas::Vector<uint> indices;
  indices.newSize(beamStateIds.size());
  //mblas::Vector<uint> indices(beamStateIds.size());
  //cerr << "indices=" << indices.Debug(2) << endl;

  mblas::copy(beamStateIds.data(),
              beamStateIds.size(),
              indices.data(),
              cudaMemcpyHostToDevice);

  CellState& cellOutStates = edOutState.GetStates();
  const CellState& cellInstates = edInState.GetStates();

  //cerr << "cellOutStates.output=" << cellOutStates.output->Debug(0) << endl;
  //cerr << "cellInstates.output=" << cellInstates.output->Debug(0) << endl;
  //cerr << "beamStateIds=" << Debug(beamStateIds, 2) << endl;
  //cerr << "indices=" << indices.Debug(2) << endl;

  //cerr << "histories=" << histories.Debug() << endl;
  //cerr << "outState1=" << outState.Debug(0) << endl;
  mblas::Assemble(*(cellOutStates.output), *(cellInstates.output), indices);
  if (cellInstates.cell->size() > 0) {
    mblas::Assemble(*(cellOutStates.cell), *(cellInstates.cell), indices);
  }
  //cerr << "edOutState.GetStates()=" << edOutState.GetStates().Debug(1) << endl;
  //cerr << "outState2=" << outState.Debug(0) << endl;

  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  decoder_->Lookup(edOutState.GetEmbeddings(), beamWords);
  //cerr << "edOutState.GetEmbeddings()=" << edOutState.GetEmbeddings().Debug(1) << endl;
  //cerr << "outState3=" << outState.Debug(0) << endl;
  //PAUSE_TIMER("AssembleBeamState");
}

void EncoderDecoder::AssembleBeamState(const std::vector<uint> newBatchIds,
                                        const mblas::Vector<uint> &d_newBatchIds,
                                        const Histories& histories,
                                        const State& inState,
                                        State& outState) const
{
  //BEGIN_TIMER("AssembleBeamState");
  std::vector<uint> beamWords;
  std::vector<uint> beamStateIds;
  for (size_t i = 0; i < histories.size(); ++i) {
    const HistoriesElementPtr &ele = histories.Get(i);
    if (ele) {
      const Hypotheses &beam = ele->GetHypotheses();
      for (const HypothesisPtr &h : beam) {
         beamWords.push_back(h->GetWord());
         beamStateIds.push_back(h->GetPrevStateIndex());
      }
    }
  }

  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  //cerr << "beamStateIds=" << Debug(beamStateIds, 2) << endl;
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "AssembleBeamState1" << endl;

  const EDState& edInState = inState.get<EDState>();
  EDState& edOutState = outState.get<EDState>();

  thread_local mblas::Vector<uint> indices;
  indices.newSize(beamStateIds.size());
  //mblas::Vector<uint> indices(beamStateIds.size());
  //cerr << "indices=" << indices.Debug(2) << endl;

  mblas::copy(beamStateIds.data(),
              beamStateIds.size(),
              indices.data(),
              cudaMemcpyHostToDevice);
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "AssembleBeamState2" << endl;

  CellState& cellOutStates = edOutState.GetStates();
  const CellState& cellInstates = edInState.GetStates();
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "AssembleBeamState3" << endl;

  //cerr << "cellOutStates1=" << cellOutStates.Debug(0) << endl;
  //cerr << "cellInstates.output=" << cellInstates.output->Debug(0) << endl;
  //cerr << "beamStateIds=" << Debug(beamStateIds, 2) << endl;
  //cerr << "indices=" << indices.Debug(2) << endl;

  //cerr << "histories=" << histories.Debug() << endl;
  //cerr << "outState1=" << outState.Debug(0) << endl;
  mblas::Assemble(*(cellOutStates.output), *(cellInstates.output), indices);
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "AssembleBeamState4" << endl;
  //cerr << "cellOutStates2=" << cellOutStates.Debug(0) << endl;

  if (cellInstates.cell->size() > 0) {
    mblas::Assemble(*(cellOutStates.cell), *(cellInstates.cell), indices);
  }
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "AssembleBeamState5" << endl;
  //cerr << "outState2=" << outState.Debug(0) << endl;

  decoder_->Lookup(edOutState.GetEmbeddings(), beamWords);
  //cerr << "outState3=" << outState.Debug(0) << endl;
  //cerr << ".GetEmbeddings()=" << .GetEmbeddings().Debug(1) << endl;
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "AssembleBeamState6" << endl;

}

}
}

