  namespace marian {

  class DistModifier {
  private:
    Ptr<Options> options_;
    bool forceDecode_{false};
    bool sampling_{false};
    std::string samplingMethod_;
    int topk_{10};
    float temperature_{1.f};

    Ptr<data::CorpusBatch> batch_;
    float invalidPathScore_;

    Expr forceBatch_;
    
  public:
    DistModifier(Ptr<Options> options, Ptr<data::CorpusBatch> batch, float invalidPathScore) :
      options_(options), forceDecode_(options_->get<bool>("force-decode", false)),
      batch_(batch), invalidPathScore_(invalidPathScore) {
      
      if(options_->hasAndNotEmpty("output-sampling")) {
        sampling_ = true;
        auto samplingOpts = options_->get<std::vector<std::string>>("output-sampling", {});
        samplingMethod_ = samplingOpts.size() > 0 ? samplingOpts[0] : "full";
        if(samplingMethod_ == "0") { // for backcompat with boolean values
          sampling_ = false;
          samplingMethod_ = "";
        } else if(samplingMethod_ == "1") { // for backcompat with boolean values
          sampling_ = true;
          samplingMethod_ = "full";
        } 
        
        if(samplingMethod_ == "full") {
          if(samplingOpts.size() > 1)
            temperature_ = std::stof(samplingOpts[1]);
        }

        if(samplingMethod_ == "topk") {
          if(samplingOpts.size() > 1)
            topk_ = std::stoi(samplingOpts[1]);
          if(samplingOpts.size() > 2)
            temperature_ = std::stof(samplingOpts[2]);
        }
      }
    }

    Expr force(Expr scores, int pos, int beamSize, std::vector<IndexType>& batchIndices) {
      // we check the last field of the batch for force-decoding content
      int dimTime = (int)batch_->back()->batchWidth();
      if(!forceDecode_ || pos >= dimTime) // nothing to force-decode, just return original scores
        return scores;

      LOG_ONCE(info, "Force-decoding with given prefixes");
      // if we get here, then we have to do force-decoding. We do this by "softly" modifying the scores and passing the 
      // result to the normal top-k/beam search. "Softly" here means we add masking terms rather than making hard selections
      // which preserves the original tensor layout.
      // This allows for beam-search and batched force-decoding with different length prefixes in a batch 
      // (way harder to do with actual index manipulation). We then return modified (masked) probabilities to the beam-search
      // which then continues as normal on the modified distribution.

      if(!forceBatch_) {
        // turn the batch into a cached tensor that lives in the computation graph
        std::vector<WordIndex> forceWords;
        for(auto& word : batch_->back()->data())
          forceWords.push_back(word.toWordIndex());
    
        int dimBatch = (int)batch_->back()->batchSize();
        forceBatch_ = scores->graph()->constant({1, dimTime, dimBatch, 1}, inits::fromVector(forceWords), Type::uint32); // [1, dimTime, dimBatch, 1]
      }

      // if we remove batch entries during decoding (finished decoding) then adjust here
      if(forceBatch_->shape()[-2] != batchIndices.size())
        forceBatch_ = index_select(forceBatch_, -2, batchIndices);

      // get vocab index and probability for force-decoded tokens for the current time step
      Expr forceIndices = slice(forceBatch_, /*axis=*/-3, pos);   // [1, 1, dimBatch, 1]
      Expr forceVals = gather(scores, /*axis=*/-1, forceIndices); // [1, 1, dimBatch, 1]

      // create dummy indices and values for beam entries other then the force-decoded value. This is required to ensure that the beam
      // does not collapse for hyps outside the forced hyps and can still do full beam-search once we finish force-decoding for a batch
      // entry. We initialize randomly (they are not going to be used anyway due to very low prob) and shift by 1 to have 0 at first postion.
      int dimVocab = scores->shape()[-1];      
      auto graph = scores->graph();
      // we start at 256 to skip over suppressed special words in SentencePiece @TODO: this should be somehow inferred.
      Expr dummyIndices = shift(graph->constant({1, 1, 1, beamSize}, inits::uniform(256.f, (float)dimVocab)), {0, 0, 0, 1}, 0.f);
      // we use a range of invalidPathScore_ to invalidPathScore_ / 2 to make sure that the probabilities stay low, but larger than invalidPathScore_ itself.
      Expr dummyVals    = shift(graph->constant({1, 1, 1, beamSize}, inits::uniform(invalidPathScore_, invalidPathScore_ / 2.f)), {0, 0, 0, 1}, 0.f);

      // here we add the force-decoded entries back into the zeroed positions
      dummyIndices = cast(cast(dummyIndices, Type::float32) + cast(forceIndices, Type::float32), Type::uint32);
      dummyVals    = dummyVals + forceVals;

      // create a tensor of the same size as the original logits, initialize with invalidPathScore and then scatter the force-decoded and 
      // dummy values into the correct positions.
      Expr forcedScores = constant_like(scores, inits::fromValue(invalidPathScore_));
      forcedScores = scatter(forcedScores, -1, dummyIndices, dummyVals);

      // for entries that have finished force-decoding (the batch has eosId as vocab id) use the original logits for the whole batch entry
      // via interpolating by a selector. In marian eosId is used for padding, so this works everywhere and eos for unfinished hyps means
      // free decoding or sampling.
      WordIndex eosId = batch_->back()->vocab()->getEosId().toWordIndex();
      auto interpol = eq(cast(forceIndices, scores->value_type()), (float)eosId);
      return interpol * scores + (1.f - interpol) * forcedScores;
    }

    Expr sample(Expr scores) {
      if(sampling_) {
        if(temperature_ != 1.f) 
          scores = scores / temperature_;
        
        if(samplingMethod_ == "full") {
          LOG_ONCE(info, "Output sampling from the full softmax distribution with temperature {}", temperature_);
          return logsoftmax(scores + constant_like(scores, inits::gumbel()));
        } else if(samplingMethod_ == "topk") {
          if(topk_ == 1)
            LOG_ONCE(info, "Output sampling with k=1 is equivalent to beam search with beam size 1");
          LOG_ONCE(info, "Output sampling via top-{} sampling with temperature {}", topk_, temperature_);
          
          Expr invalidLogits = constant_like(scores, inits::fromValue(invalidPathScore_));
          
          // select top-k values
          Expr val, idx;
          std::tie(val, idx) = topk(scores, topk_, /*axis=*/-1, /*descending=*/true);
          
          // Add Gumbel noise to top-k values only and compute logsoftmax, used for argmax sampling later in beam-search
          Expr gumbelVal = logsoftmax(val + constant_like(val, inits::gumbel()));

          // Scatter gumbelled values back into logits to fill with usable values
          return scatter(invalidLogits, -1, idx, gumbelVal);
        } else {
          ABORT("Unknown sampling method: {}", samplingMethod_);
        }
      } else { // no sampling
        return scores;
      }
    }

  };

  }