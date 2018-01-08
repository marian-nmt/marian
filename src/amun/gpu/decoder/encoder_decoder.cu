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
    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "DecodeAsyncInternal1" << endl;
    //std::cerr << "histories=" << histories.Debug(1) << std::endl;

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

    //if (histories.NumActive() < 8) {
    if (unsigned numSentToGet = SentencesToGet(histories)) {
      //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      //cerr << "DecodeAsyncInternal6" << endl;
      //std::cerr << "histories6=" << histories.Debug(1) << std::endl;

      //InitBatch(histories, sentenceLengths, sourceContext, SCU, *state);
      TopupBatch(histories, numSentToGet, sentenceLengths, sourceContext, SCU, *nextState, *state);
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

    LOG(progress)->info("  Step {} took {} sentences {} hypos {}", step++, timerStep.format(5, "%w"), histories.NumActive(), histories.GetTotalBeamSize());
  }
}

void EncoderDecoder::InitBatch(Histories &histories,
                                mblas::Vector<unsigned> &sentenceLengths,
                                mblas::Matrix &sourceContext,
                                mblas::Matrix &SCU,
                                State &state)
{
  ///*
  unsigned miniBatch = god_.Get<unsigned>("mini-batch");

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

  EncOutGPU &encOutGPU = encOut->Get<EncOutGPU>();

  sourceContext.swap(encOutGPU.GetSourceContext());
  SCU.swap(encOutGPU.GetSCU());

  CellState& cellState = state.get<EDState>().GetStates();
  CellState& origCellState = encOutGPU.GetCellState();
  cellState.output->swap(*origCellState.output);
  cellState.cell->swap(*origCellState.cell);

  histories.Init(newSentences);
  //cerr << "histories=" << histories.Debug() << endl;

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

  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "TopupBatch2" << endl;
  //cerr << "numSentToGet=" << numSentToGet << endl;
  //cerr << "histories orig=" << histories.Debug() << endl;

  std::vector<BufferOutput> newSentences;
  encDecBuffer_.Get(numSentToGet, newSentences);
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "TopupBatch3" << endl;
  //cerr << "newSentences=" << newSentences.size() << endl;

  histories.StartTopup();

  // update existing batch
  for (unsigned i = 0; i < newSentences.size(); ++i) {
    //cerr << "TopupBatch4" << endl;
    const BufferOutput &eleSent = newSentences[i];
    const SentencePtr &sentence = eleSent.GetSentence();

    histories.Topup(new HistoriesElement(sentence, histories.NormalizeScore()));

  }
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "TopupBatch5" << endl;

  if (histories.NumActive() == 0) {
    PAUSE_TIMER("TopupBatch");
    return;
  }

  // histories is const from here on
  //cerr << "histories=" << histories.Debug() << endl;

  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "TopupBatch6" << endl;
  //cerr << "histories=" << histories.Debug() << endl;

  std::vector<unsigned> newBatchIds, oldBatchIds, newSentenceLengths, newHypoIds, oldHypoIds;
  histories.BatchIds(newBatchIds, oldBatchIds, newSentenceLengths, newHypoIds, oldHypoIds);

  mblas::Vector<unsigned> d_newBatchIds(newBatchIds);
  mblas::Vector<unsigned> d_oldBatchIds(oldBatchIds);
  mblas::Vector<unsigned> d_newSentenceLengths(newSentenceLengths);
  mblas::Vector<unsigned> d_newHypoIds(newHypoIds);
  mblas::Vector<unsigned> d_oldHypoIds(oldHypoIds);

  //cerr << "newBatchIds=" << Debug(newBatchIds, 2) << endl;
  //cerr << "oldBatchIds=" << Debug(oldBatchIds, 2) << endl;
  //cerr << "newHypoIds=" << Debug(newHypoIds, 2) << endl;
  //cerr << "oldHypoIds=" << Debug(oldHypoIds, 2) << endl;
  //cerr << "newSentenceLengths=" << Debug(newSentenceLengths, 2) << endl;

  //cerr << "4state=" << state.Debug() << endl;
  AssembleBeamStateTopup(histories, nextState, oldHypoIds, state);
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "TopupBatch7" << endl;
  //cerr << "5state=" << state.Debug() << endl;

  if (newSentences.size()) {
    unsigned maxLength =  histories.MaxLength();
    //cerr << "maxLength=" << maxLength << endl;

    UpdateSentenceLengths(d_newBatchIds, d_newSentenceLengths, sentenceLengths);
    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "TopupBatch8" << endl;

    // source context
    //cerr << "1sourceContext=" << sourceContext.Debug() << endl;
    ResizeMatrix3(sourceContext, {0, maxLength}, d_oldBatchIds);
    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "2sourceContext=" << sourceContext.Debug() << endl;
    //cerr << "TopupBatch9" << endl;

    AddNewSourceContext(sourceContext, newBatchIds, newSentences);
    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "TopupBatch10" << endl;
    //cerr << "3sourceContext=" << sourceContext.Debug() << endl;

    //cerr << "1SCU=" << SCU.Debug() << endl;
    BeginSentenceStateTopup(histories, sourceContext, state, SCU, newSentences, d_oldBatchIds, newBatchIds, newHypoIds, d_newHypoIds);
    //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    //cerr << "TopupBatch11" << endl;
    //cerr << "histories new=" << histories.Debug() << endl;
    //cerr << "2SCU=" << SCU.Debug() << endl;
    //cerr << "6state=" << state.Debug() << endl;

    /*
    for (unsigned i = 0; i < newSentences.size(); ++i) {
      //cerr << "TopupBatch12" << endl;
      BufferOutput &eleSent = newSentences[i];
      eleSent.Release();
    }
    */
  }

  //LOG(progress)->info("Topup took {} new {} histories {}", timer.format(5, "%w"), newSentences.size(), histories.NumActive());
  //cerr << endl;
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

  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "BeginSentenceState1" << endl;

  //cerr << "1state=" << state.Debug() << endl;
  decoder_->EmptyStateTopup(edState.GetStates(),
                            sourceContext,
                            SCU,
                            newSentences,
                            d_oldBatchIds,
                            newBatchIds,
                            newHypoIds);
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "BeginSentenceState2" << endl;
  //cerr << "2state=" << state.Debug() << endl;

  decoder_->EmptyEmbeddingTopup(edState.GetEmbeddings(), histories.GetTotalBeamSize(), d_newHypoIds);
  //PAUSE_TIMER("BeginSentenceState");
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "BeginSentenceState3" << endl;
  //cerr << "3state=" << state.Debug() << endl;
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

void EncoderDecoder::AssembleBeamState(const Histories& histories,
                                        const State& inState,
                                        State& outState) const
{
  //BEGIN_TIMER("AssembleBeamState");
  std::vector<unsigned> beamWords, beamStateIds;
  histories.AssembleInfo(beamWords, beamStateIds);
  /*
  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  cerr << "AssembleBeamState1" << endl;
  cerr << "histories=" << histories.Debug(2) << endl;
  cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  cerr << "beamStateIds=" << Debug(beamStateIds, 2) << endl;
  */

  const EDState& edInState = inState.get<EDState>();
  EDState& edOutState = outState.get<EDState>();

  thread_local mblas::Vector<unsigned> indices;
  indices.newSize(beamStateIds.size());
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

void EncoderDecoder::AssembleBeamStateTopup(const Histories& histories,
                                        const State& inState,
                                        const mblas::Vector<unsigned> &d_oldHypoIds,
                                        State& outState) const
{
  //BEGIN_TIMER("AssembleBeamState");
  std::vector<unsigned> beamWords, beamStateIds;
  histories.AssembleInfo(beamWords, beamStateIds);
  /*
  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  cerr << "AssembleBeamState1" << endl;
  cerr << "histories=" << histories.Debug(2) << endl;
  */
  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  //cerr << "beamStateIds=" << Debug(beamStateIds, 2) << endl;

  unsigned numHypos = histories.GetTotalBeamSize();

  const EDState& edInState = inState.get<EDState>();
  EDState& edOutState = outState.get<EDState>();

  thread_local mblas::Vector<unsigned> indices;
  indices.newSize(beamStateIds.size());
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
  mblas::AssembleTopup(*(cellOutStates.output), *(cellInstates.output), indices, numHypos, d_oldHypoIds);
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "AssembleBeamState4" << endl;
  //cerr << "cellOutStates2=" << cellOutStates.Debug(0) << endl;

  if (cellInstates.cell->size() > 0) {
    mblas::AssembleTopup(*(cellOutStates.cell), *(cellInstates.cell), indices, numHypos, d_oldHypoIds);
  }
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "AssembleBeamState5" << endl;
  //cerr << "outState2=" << outState.Debug(0) << endl;

  decoder_->LookupTopup(edOutState.GetEmbeddings(), beamWords, histories, d_oldHypoIds);
  //cerr << "outState3=" << outState.Debug(0) << endl;
  //cerr << ".GetEmbeddings()=" << .GetEmbeddings().Debug(1) << endl;
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "AssembleBeamState6" << endl;

}

unsigned EncoderDecoder::SentencesToGet(const Histories& histories)
{
  /*
  unsigned ret = god_.Get<unsigned>("mini-batch") - histories.NumActive();
  return ret;
  */
  ///*
  BEGIN_TIMER("SentencesToGet");

  const unsigned MIN_ACTIVE = 120;

  unsigned beamSize = god_.Get<unsigned>("beam-size");
  unsigned numHypos = histories.GetTotalBeamSize();

  unsigned start = std::max(MIN_ACTIVE, histories.NumActive());

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
  else if (histories.NumActive() < MIN_ACTIVE) {
    ret = MIN_ACTIVE - histories.NumActive();
  }
  else {
    ret = 0;
  }

  PAUSE_TIMER("SentencesToGet");
  return ret;
  //*/
}

}
}

