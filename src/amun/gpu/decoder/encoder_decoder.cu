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
        unsigned tab,
        const Weights& model,
        const Search &search)
  : Scorer(god, name, config, tab, search),
    model_(model),
    encoder_(new Encoder(model_, config)),
    decoder_(new Decoder(god, model_, config)),
    encDecBuffer_(god.Get<unsigned>("encoder-buffer-size"))
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

  SetTensorCore();

  EncOutPtr encOut(new EncOutGPU(source));

  if (source->size()) {
    BEGIN_TIMER("Encode.Calc");
    encoder_->Encode(encOut, tab_);

    EncOutGPU &encOutGPU = encOut->Get<EncOutGPU>();
    //auto aligner = decoder_->GetAligner();
    decoder_->GetAligner().Init(encOutGPU.GetSourceContext(), encOutGPU.GetSCU());
    decoder_->GetHiddenRNN().InitializeState(encOutGPU.GetCellState(),
                                            encOutGPU.GetSourceContext(),
                                            encOutGPU.GetSentences().size(),
                                            encOutGPU.GetSentenceLengths());
    PAUSE_TIMER("Encode.Calc");
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

unsigned EncoderDecoder::GetVocabSize() const {
  return decoder_->GetVocabSize();
}

void EncoderDecoder::Filter(const std::vector<unsigned>& filterIds) {
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
  SetTensorCore();

  unsigned maxBeamSize = god_.Get<unsigned>("beam-size");

  unsigned miniBatch = god_.Get<unsigned>("mini-batch");
  unsigned decodingBatch = god_.Get<unsigned>("decoding-mini-batch");
  if (decodingBatch) {
    miniBatch = decodingBatch;
  }

  Histories histories(search_.NormalizeScore());

  mblas::Vector<unsigned> sentenceLengths;
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

    decoder_->Decode(nextState->get<EDState>().GetStates(),
                    state->get<EDState>().GetStates(),
                    state->get<EDState>().GetEmbeddings(),
                    histories,
                    god_.UseFusedSoftmax(),
                    sourceContext,
                    SCU,
                    sentenceLengths);

    histories.SetNewBeamSize(maxBeamSize);
    CalcBeam(search_.GetBestHyps(), histories, *state, *nextState, search_.GetFilterIndices());

    //if (histories.NumActive() < 8) {
    if (unsigned numSentToGet = SentencesToGet(histories)) {
      BEGIN_TIMER("TopupBatch.outer");
      //InitBatch(histories, sentenceLengths, sourceContext, SCU, *state);
      TopupBatch(histories, numSentToGet, sentenceLengths, sourceContext, SCU, *nextState, *state);
      PAUSE_TIMER("TopupBatch.outer");
    }
    else {
      AssembleBeamState(histories, *nextState, *state);
    }

    LOG(progress)->info("  Step {} took {} sentences {} hypos {} firsts {}",
                        step++,
                        timerStep.format(5, "%w"),
                        histories.NumActive(),
                        histories.GetTotalBeamSize(),
                        histories.NumFirsts());
  }
}

void EncoderDecoder::InitBatch(Histories &histories,
                                mblas::Vector<unsigned> &sentenceLengths,
                                mblas::Matrix &sourceContext,
                                mblas::Matrix &SCU,
                                State &state)
{
  unsigned miniBatch = god_.Get<unsigned>("mini-batch");

  std::vector<BufferOutput> newSentences;
  encDecBuffer_.Get(miniBatch, newSentences);

  //vector<unsigned> batchIds = AddToBatch(newSentences, sentences, histories, sentenceLengths, sourceContext);

  if (newSentences.size() == 0) {
    return;
  }

  const EncOutPtr &encOut = newSentences.front().GetEncOut();
  assert(encOut);

  EncOutGPU &encOutGPU = encOut->Get<EncOutGPU>();

  sentenceLengths.swap(encOutGPU.GetSentenceLengths());
  sourceContext.swap(encOutGPU.GetSourceContext());
  SCU.swap(encOutGPU.GetSCU());

  CellState& cellState = state.get<EDState>().GetStates();
  CellState& origCellState = encOutGPU.GetCellState();
  cellState.output->swap(*origCellState.output);
  cellState.cell->swap(*origCellState.cell);

  histories.Init(newSentences);

  BeginSentenceState(histories, sourceContext, sentenceLengths, state, SCU);

  return;
}

void EncoderDecoder::TopupBatch(Histories &histories,
                                unsigned numSentToGet,
                                mblas::Vector<unsigned> &sentenceLengths,
                                mblas::Matrix &sourceContext,
                                mblas::Matrix &SCU,
                                State &nextState,
                                State &state)
{
  BEGIN_TIMER("TopupBatch");
  boost::timer::cpu_timer timer;

  std::vector<BufferOutput> newSentences;
  encDecBuffer_.Get(numSentToGet, newSentences);

  histories.StartTopup();

  // update existing batch
  for (unsigned i = 0; i < newSentences.size(); ++i) {
    //cerr << "TopupBatch4" << endl;
    const BufferOutput &eleSent = newSentences[i];
    const SentencePtr &sentence = eleSent.GetSentence();

    histories.Topup(new HistoriesElement(sentence, histories.NormalizeScore()));

  }

  if (histories.NumActive() == 0) {
    PAUSE_TIMER("TopupBatch");
    return;
  }

  std::vector<unsigned> newBatchIds, oldBatchIds, newSentenceLengths, newHypoIds, oldHypoIds;
  BEGIN_TIMER("TopupBatch.1");
  histories.BatchIds(newBatchIds, oldBatchIds, newSentenceLengths, newHypoIds, oldHypoIds);
  PAUSE_TIMER("TopupBatch.1");
  
  thread_local mblas::Vector<unsigned> d_newBatchIds, d_oldBatchIds, d_newSentenceLengths, d_newHypoIds, d_oldHypoIds;
  d_newBatchIds.copyFrom(newBatchIds);
  d_oldBatchIds.copyFrom(oldBatchIds);
  d_newSentenceLengths.copyFrom(newSentenceLengths);
  d_newHypoIds.copyFrom(newHypoIds);
  d_oldHypoIds.copyFrom(oldHypoIds);


  BEGIN_TIMER("TopupBatch.2");
  AssembleBeamStateTopup(histories, nextState, oldHypoIds, state);
  PAUSE_TIMER("TopupBatch.2");
  
  if (newSentences.size()) {
    unsigned maxLength =  histories.MaxLength();

    BEGIN_TIMER("TopupBatch.3");
    UpdateSentenceLengths(d_newBatchIds, d_newSentenceLengths, sentenceLengths);
    PAUSE_TIMER("TopupBatch.3");
    
    BEGIN_TIMER("TopupBatch.4");
    ResizeMatrix3(sourceContext, {0, maxLength}, d_oldBatchIds);
    PAUSE_TIMER("TopupBatch.4");
    
    BEGIN_TIMER("TopupBatch.5");
    AddNewSourceContext(sourceContext, newBatchIds, newSentences);
    PAUSE_TIMER("TopupBatch.5");
    
    BEGIN_TIMER("TopupBatch.6");
    BeginSentenceStateTopup(histories, sourceContext, state, SCU, newSentences, d_oldBatchIds, newBatchIds, newHypoIds, d_newHypoIds);
    PAUSE_TIMER("TopupBatch.6");
  }

  //LOG(progress)->info("Topup took {} new {} histories {}", timer.format(5, "%w"), newSentences.size(), histories.NumActive());
  PAUSE_TIMER("TopupBatch");
}

void EncoderDecoder::BeginSentenceState(const Histories& histories,
                                        const mblas::Matrix &sourceContext,
                                        const mblas::Vector<unsigned> &sentenceLengths,
                                        State& state,
                                        mblas::Matrix& SCU) const
{
  //BEGIN_TIMER("BeginSentenceState");
  EDState& edState = state.get<EDState>();

  decoder_->EmptyEmbedding(edState.GetEmbeddings(), histories.NumActive());
  //PAUSE_TIMER("BeginSentenceState");
}

void EncoderDecoder::BeginSentenceStateTopup(const Histories& histories,
                                        const mblas::Matrix &sourceContext,
                                        State& state,
                                        mblas::Matrix& SCU,
                                        const std::vector<BufferOutput> &newSentences,
                                        const mblas::Vector<unsigned> &d_oldBatchIds,
                                        const std::vector<unsigned> &newBatchIds,
                                        const std::vector<unsigned> &newHypoIds,
                                        const mblas::Vector<unsigned> &d_newHypoIds) const
{
  //BEGIN_TIMER("BeginSentenceState");
  EDState& edState = state.get<EDState>();
  unsigned batchSize = histories.NumActive();

  decoder_->EmptyStateTopup(edState.GetStates(),
                            sourceContext,
                            SCU,
                            newSentences,
                            d_oldBatchIds,
                            newBatchIds,
                            newHypoIds);

  decoder_->EmptyEmbeddingTopup(edState.GetEmbeddings(), histories.GetTotalBeamSize(), d_newHypoIds);
}

void EncoderDecoder::CalcBeam(BestHypsBase &bestHyps,
                      Histories& histories,
                      State& state,
                      State& nextState,
                      const Words &filterIndices)
{
  Hypotheses prevHypos = histories.GetSurvivors();

  histories.StartCalcBeam();
  bestHyps.CalcBeam(prevHypos, *this, filterIndices, histories);
  histories.Add(god_);
}

void EncoderDecoder::AssembleBeamState(const Histories& histories,
                                        const State& inState,
                                        State& outState) const
{
  //BEGIN_TIMER("AssembleBeamState");
  std::vector<unsigned> beamWords, beamStateIds;
  histories.AssembleInfo(beamWords, beamStateIds);

  const EDState& edInState = inState.get<EDState>();
  EDState& edOutState = outState.get<EDState>();

  thread_local mblas::Vector<unsigned> indices;
  indices.copyFrom(beamStateIds);

  CellState& cellOutStates = edOutState.GetStates();
  const CellState& cellInstates = edInState.GetStates();

  mblas::Assemble(*(cellOutStates.output), *(cellInstates.output), indices);
  if (cellInstates.cell->size() > 0) {
    mblas::Assemble(*(cellOutStates.cell), *(cellInstates.cell), indices);
  }

  decoder_->Lookup(edOutState.GetEmbeddings(), beamWords);
  //PAUSE_TIMER("AssembleBeamState");
}

void EncoderDecoder::AssembleBeamStateTopup(const Histories& histories,
                                        const State& inState,
                                        const mblas::Vector<unsigned> &d_oldHypoIds,
                                        State& outState) const
{
  //BEGIN_TIMER("AssembleBeamState");
  std::vector<unsigned> beamWords, beamStateIds;
  histories.AssembleInfo(beamWords, beamStateIds);
  unsigned numHypos = histories.GetTotalBeamSize();

  const EDState& edInState = inState.get<EDState>();
  EDState& edOutState = outState.get<EDState>();

  thread_local mblas::Vector<unsigned> indices;
  indices.copyFrom(beamStateIds);

  CellState& cellOutStates = edOutState.GetStates();
  const CellState& cellInstates = edInState.GetStates();
  mblas::AssembleTopup(*(cellOutStates.output), *(cellInstates.output), indices, numHypos, d_oldHypoIds);

  if (cellInstates.cell->size() > 0) {
    mblas::AssembleTopup(*(cellOutStates.cell), *(cellInstates.cell), indices, numHypos, d_oldHypoIds);
  }

  decoder_->LookupTopup(edOutState.GetEmbeddings(), beamWords, histories, d_oldHypoIds);
}

unsigned EncoderDecoder::SentencesToGet(const Histories& histories)
{
  //return histories.NumInactive(); 
  ///*
  if (histories.size() < 8 || histories.NumActive() < histories.size() / 2) {
    return histories.NumInactive();
  }
  else {
    return 0;
  }
  //*/
  /*
  BEGIN_TIMER("SentencesToGet");

  unsigned minActive = (histories.size() > 8) ? histories.size() - 8 : 1;

  unsigned beamSize = god_.Get<unsigned>("beam-size");
  unsigned numHypos = histories.GetTotalBeamSize();

  unsigned start = std::max(minActive, histories.NumActive());

  unsigned okNum = 0;
  for (unsigned currSize = start; currSize <= histories.size(); ++currSize) {
    unsigned numNew = currSize - histories.NumActive();
    if ((numHypos + numNew) % 8 == 0) {
      if ((histories.NumActive() + numNew) % 8 == 0) {
        PAUSE_TIMER("SentencesToGet");
        return numNew;
      }
      else if (okNum == 0) {
        okNum = currSize;
      }
    }
  }

  unsigned ret;
  if (okNum) {
    ret = okNum - histories.NumActive();
  }
  else if (histories.NumActive() < minActive) {
    ret = minActive - histories.NumActive();
  }
  else {
    ret = 0;
  }

  PAUSE_TIMER("SentencesToGet");
  return ret;
  */
}

void EncoderDecoder::SetTensorCore()
{
#if CUDA_VERSION >= 9000
  if (god_.UseTensorCores()) {
    //cerr << "using tensor cores" << endl;
    cublasHandle_t handle = mblas::CublasHandler::GetHandle();
    cublasStatus_t stat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("cublasSetMathMode failed\n");
      abort();
    }
  }
#endif

}

}
}

