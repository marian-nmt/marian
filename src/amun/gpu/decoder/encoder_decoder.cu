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

  Sentences sentences(miniBatch);
  mblas::Vector<uint> sentenceLengths;
  mblas::Matrix sourceContext, SCU;
  StatePtr state, nextState;
  Hypotheses prevHyps;

  bool hasSentences = FetchBatch(sentences, histories, sentenceLengths, sourceContext, SCU, state, nextState, prevHyps);

  unsigned step = 0;
  while (hasSentences && histories.GetNumActive()) {
    boost::timer::cpu_timer timerStep;

    const EDState& edstate = state->get<EDState>();
    EDState& ednextState = nextState->get<EDState>();

    decoder_->Decode(ednextState.GetStates(),
                     edstate.GetStates(),
                     edstate.GetEmbeddings(),
                     histories,
                     god_.UseFusedSoftmax(),
                     sourceContext,
                     SCU,
                     sentenceLengths);

    histories.SetNewBeamSize(maxBeamSize);

    unsigned numPrevHyps = prevHyps.size();
    size_t survivors = CalcBeam(search_.GetBestHyps(), histories, prevHyps, *state, *nextState, search_.GetFilterIndices());

    if (survivors == 0) {
      hasSentences = FetchBatch(sentences, histories, sentenceLengths, sourceContext, SCU, state, nextState, prevHyps);
    }

    LOG(progress)->info("  Step {} took {} sentences {} prevHypos {} survivors {}", step++, timerStep.format(5, "%w"), histories.GetNumActive(), numPrevHyps, survivors);
  }

}

bool EncoderDecoder::FetchBatch(Sentences &sentences,
                                Histories &histories,
                                mblas::Vector<uint> &sentenceLengths,
                                mblas::Matrix &sourceContext,
                                mblas::Matrix &SCU,
                                StatePtr &state,
                                StatePtr &nextState,
                                Hypotheses &prevHyps)
{
  ///*
  uint miniBatch = god_.Get<uint>("mini-batch");

  std::vector<BufferOutput> newSentences;
  encDecBuffer_.Get(miniBatch, newSentences);

  //vector<unsigned> batchIds = AddToBatch(newSentences, sentences, histories, sentenceLengths, sourceContext);

  if (newSentences.size() == 0) {
    return false;
  }

  sentences.ResetAll();
  for (size_t i = 0; i < newSentences.size(); ++i) {
    const BufferOutput &ele = newSentences[i];
    const SentencePtr &sentence = ele.GetSentence();
    sentences.Set(i, sentence);
  }
  sentences.RecalcMaxLength();

  const EncOutPtr &encOut = newSentences.front().encOut;
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

  sentenceLengths = encOut->Get<EncOutGPU>().GetSentenceLengths();
  sourceContext = encOut->Get<EncOutGPU>().GetSourceContext();

  state.reset(NewState());
  nextState.reset(NewState());

  histories.Init(newSentences);

  BeginSentenceState(histories.GetNumActive(), sourceContext, sentenceLengths, *state, SCU);

  prevHyps = histories.GetFirstHyps();

  return true;
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

  AssembleBeamState(nextState, survivors, state);

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
  //cerr << "indices=" << mblas::Debug(indices, 2) << endl;

  CellState& outstates = edNextState.GetStates();
  const CellState& instates = edState.GetStates();

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

//////////////////////////////////////////////////////////////////////
//helper fn
size_t FindNextEmptyIndex(size_t nextBatchInd,
                        Histories &histories)
{
  while(nextBatchInd < histories.size()) {
    const HistoriesElementPtr &ele = histories.Get(nextBatchInd);
    if (ele == nullptr) {
      return nextBatchInd;
    }
    ++nextBatchInd;
  }

  assert(false);
  return 9999999;
}

////////////////////////////////////////////////////////////////////////

vector<unsigned> EncoderDecoder::AddToBatch(const std::vector<BufferOutput> &newSentences,
                                Sentences &sentences,
                                Histories &histories,
                                mblas::Vector<uint> &sentenceLengths,
                                mblas::Matrix &sourceContext)
{
  cerr << "sentenceLengths=" << sentenceLengths.Debug() << endl;

  vector<uint> newSentenceLengths(newSentences.size());
  vector<uint> newBatchIds(newSentences.size());

  // update existing batch
  size_t nextBatchInd = 0;
  for (size_t i = 0; i < newSentences.size(); ++i) {
    const BufferOutput &eleSent = newSentences[i];
    const SentencePtr &sentence = eleSent.GetSentence();

    // work out offset in existing batch
    size_t batchInd = FindNextEmptyIndex(nextBatchInd, histories);
    newBatchIds[i] = batchInd;

    // sentences
    sentences.Set(batchInd, sentence);

    // sentence lengths
    newSentenceLengths[i] = sentence->size();

    // histories
    HistoriesElementPtr &eleHist = histories.Get(nextBatchInd);
    assert(eleHist == nullptr);
    eleHist.reset(new HistoriesElement(sentence, histories.NormalizeScore()));

    nextBatchInd = batchInd + 1;
  }

  sentences.RecalcMaxLength();

  // update gpu data
  mblas::Vector<uint> d_newSentenceLengths(newSentenceLengths);
  mblas::Vector<uint> d_newBatchIds(newBatchIds);
  UpdateSentenceLengths(d_newSentenceLengths, d_newBatchIds, sentenceLengths);

  unsigned maxLength = sentences.GetMaxLength();

  cerr << "newSentences=" << newSentences.size() << endl;
  cerr << "sourceContext1=" << sourceContext.Debug() << endl;
  cerr << "newMaxLength=" << maxLength << endl;

  // source context
  EnlargeMatrix(sourceContext, 0, maxLength);
  cerr << "sourceContext2=" << sourceContext.Debug() << endl;

  for (size_t i = 0; i < newSentences.size(); ++i) {
    const BufferOutput &eleSent = newSentences[i];
    //size_t sentenceInd = eleSent.sentenceInd;
    const EncOutPtr &encOut = eleSent.encOut;
    const mblas::Matrix &newSourceContext = encOut->Get<EncOutGPU>().GetSourceContext();
    cerr << "newSourceContext=" << newSourceContext.Debug() << endl;

    //AddNewData(d_newBatchIds,  sourceContext);
  }

  return newBatchIds;
}

}
}

