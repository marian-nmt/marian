#pragma once

#include <thread>

#include "common/definitions.h"
#include "3rd_party/threadpool.h"

namespace marian {

class Reporter {
  public:
    Ptr<Config> options_;

    float costSum{0};
    size_t epochs{1};

    size_t samples{0};
    size_t wordsDisp{0};
    size_t batches{0};

    boost::timer::cpu_timer timer;

  public:
    Reporter(Ptr<Config> options) : options_(options) {}

    void update(float cost, Ptr<data::CorpusBatch> batch) {
      static std::mutex sMutex;
      std::lock_guard<std::mutex> guard(sMutex);

      costSum += cost;
      samples += batch->size();
      wordsDisp += batch->words();
      batches++;
      //if(options.get<size_t>("after-batches")
      //   && batches >= options.get<size_t>("after-batches"))
      //  break;

      if(batches % options_->get<size_t>("disp-freq") == 0) {
        std::stringstream ss;
        ss << "Ep. " << epochs
           << " : Up. " << batches
           << " : Sen. " << samples
           << " : Cost " << std::fixed << std::setprecision(2)
                         << costSum / options_->get<size_t>("disp-freq")
           << " : Time " << timer.format(2, "%ws");

        float seconds = std::stof(timer.format(5, "%w"));
        float wps = wordsDisp /   (float)seconds;

        ss << " : " << std::fixed << std::setprecision(2)
           << wps << " words/s";

        LOG(info) << ss.str();

        timer.start();
        costSum = 0;
        wordsDisp = 0;
      }
    }
};

class GraphGroup {
  protected:
    Ptr<Config> options_;
    Ptr<Reporter> reporter_;
    Ptr<OptimizerBase> opt_;

    std::vector<Ptr<ExpressionGraph>> graphs_;

  public:
    GraphGroup(Ptr<Config> options)
    : options_(options), opt_(Optimizer(options)) { }

    virtual void update(Ptr<data::CorpusBatch>) = 0;

    virtual void setReporter(Ptr<Reporter> reporter) {
      reporter_ = reporter;
    }

    virtual void save() = 0;
};


template <class Builder>
class AsyncGraphGroup : public GraphGroup {
  private:
    Ptr<Builder> builder_;

    std::vector<size_t> devices_;
    ThreadPool pool_;

    std::vector<Ptr<ExpressionGraph>> graphs_;

    std::mutex sync_;

    Tensor params_;
    Ptr<TensorAllocator> paramsAlloc_;

    Tensor grads_;
    Ptr<TensorAllocator> gradsAlloc_;

    void fetchParams(Tensor oldParams) {
      if(graphs_.size() < 2)
        return;

      // @TODO read guard on parameters
      std::lock_guard<std::mutex> guard(sync_);
      oldParams->copyFrom(params_);
    }

    void pushGradients(Tensor newGrads) {
      if(graphs_.size() < 2) {
        opt_->update(graphs_[0]);
      }
      else {
        std::lock_guard<std::mutex> guard(sync_);
        grads_->copyFrom(newGrads);
        opt_->update(params_, grads_);
      }
    }

    void execute(Ptr<data::CorpusBatch> batch) {
      static bool first = true;
      if(first && graphs_.size() > 1) {
        // initialize the paramters
        for(auto graph : graphs_) {
          builder_->build(graph, batch);
          graph->forward();
        }

        if(!params_) {
          paramsAlloc_ = New<TensorAllocator>(graphs_[0]->getDevice());

          int totalSize = graphs_[0]->params().vals()->size();
          paramsAlloc_->reserveExact(totalSize);
          paramsAlloc_->allocate(params_, {1, totalSize});
        }

        if(!grads_) {
          gradsAlloc_ = New<TensorAllocator>(graphs_[0]->getDevice());

          int totalSize = graphs_[0]->params().vals()->size();
          gradsAlloc_->reserveExact(totalSize);
          gradsAlloc_->allocate(grads_, {1, totalSize});
        }

        params_->copyFrom(graphs_[0]->params().vals());
        first = false;
      }

      auto task = [this](Ptr<data::CorpusBatch> batch) {
        static size_t i = 0;
        thread_local Ptr<ExpressionGraph> graph;
        if(!graph) {
          std::lock_guard<std::mutex> lock(sync_);
          graph = graphs_[i++];
        }

        builder_->build(graph, batch);

        fetchParams(graph->params().vals());

        graph->forward();
        float cost = graph->topNode()->scalar();
        graph->backward();

        pushGradients(graph->params().grads());

        if(reporter_) {
          std::lock_guard<std::mutex> guard(sync_);
          reporter_->update(cost, batch);
          if(reporter_->batches % options_->get<size_t>("save-freq") == 0)
            this->save();
        }
      };

      pool_.enqueue(task, batch);
    }

  public:
    AsyncGraphGroup(Ptr<Config> options)
     : GraphGroup(options),
       builder_{New<Builder>(options_)},
       devices_{options_->get<std::vector<size_t>>("device")},
       pool_{devices_.size(), devices_.size() } {

      for(auto device : devices_) {
        graphs_.emplace_back(New<ExpressionGraph>());
        graphs_.back()->setDevice(device);
        graphs_.back()->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      }
    }

    void update(Ptr<data::CorpusBatch> batch) {
      execute(batch);
    }

    void save() {
      std::lock_guard<std::mutex> guard(sync_);
      if(options_->get<bool>("overwrite")) {
        std::string name = options_->get<std::string>("model") + ".npz";
        builder_->save(graphs_[0], name);
      }
      else {
        std::string name = options_->get<std::string>("model")
          + "." + std::to_string(reporter_->batches) + ".npz";
        builder_->save(graphs_[0], name);
      }
    }
};


template <class Builder>
class SyncGraphGroup : public GraphGroup {
  private:
    Ptr<Builder> builder_;
    std::vector<Ptr<data::CorpusBatch>> batches_;

    bool first_{true};

    void accumulateGradients(Ptr<ExpressionGraph> master,
                             std::vector<Ptr<ExpressionGraph>> graphs) {
      if(graphs_.size() < 2) {
        return;
      }

      Tensor grads = master->params().grads();
      Tensor tempGrads;
      master->tensor(tempGrads, grads->shape());

      for(auto graph : graphs) {
        if(graph != master) {
          Tensor remoteGrads = graph->params().grads();
          tempGrads->copyFrom(remoteGrads);
          Element(_1 += _2, grads, tempGrads);
        }
      }

      float denom = graphs_.size();
      Element(_1 /= denom, grads);
    }

    void distributeParameters(Ptr<ExpressionGraph> master,
                              std::vector<Ptr<ExpressionGraph>> graphs) {
      if(graphs_.size() < 2)
        return;

      Tensor params = master->params().vals();
      for(auto graph : graphs) {
        if(graph != master) {
          graph->params().vals()->copyFrom(params);
        }
      }
    }

    void execute() {
      if(first_) {
        for(auto graph : graphs_) {
          builder_->build(graph, batches_[0]);
          graph->forward();
        }
        distributeParameters(graphs_[0], graphs_);
        first_ = false;
      }

      auto task = [this](int i,
                         Ptr<data::CorpusBatch> batch) {
        thread_local int j = -1;
        if(j == -1)
          j = i;
        auto localGraph = this->graphs_[j];

        builder_->build(localGraph, batch);
        localGraph->forward();
        float cost = localGraph->topNode()->scalar();
        localGraph->backward();

        if(reporter_) {
          reporter_->update(cost, batch);
          if(reporter_->batches % options_->get<size_t>("save-freq") == 0)
            this->save();
        }
      };

      {
        size_t workers = graphs_.size();
        ThreadPool pool(workers, workers);

        for(int i = 0; i < batches_.size(); ++i)
          pool.enqueue(task, i % (int)workers, batches_[i]);
      }
      accumulateGradients(graphs_[0], graphs_);
      opt_->update(graphs_[0]);
      distributeParameters(graphs_[0], graphs_);

      batches_.clear();
    }

  public:
    SyncGraphGroup(Ptr<Config> options)
     : GraphGroup(options),
       builder_{New<Builder>(options_)} {

      auto devices = options_->get<std::vector<size_t>>("device");
      size_t workers = devices.size();

      for(auto device : devices) {
        graphs_.emplace_back(New<ExpressionGraph>());
        graphs_.back()->setDevice(device);
        graphs_.back()->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      }

    }

    ~SyncGraphGroup() {
      execute();
    }

    void update(Ptr<data::CorpusBatch> batch) {
      batches_.push_back(batch);
      if(batches_.size() == graphs_.size())
        execute();
    }

    void save() {
      if(options_->get<bool>("overwrite")) {
        std::string name = options_->get<std::string>("model") + ".npz";
        builder_->save(graphs_[0], name);
      }
      else {
        std::string name = options_->get<std::string>("model")
          + "." + std::to_string(reporter_->batches) + ".npz";
        builder_->save(graphs_[0], name);
      }
    }
};

}
