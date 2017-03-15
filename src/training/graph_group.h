#pragma once
#define GRAD_DROP true
#define SPARSE_PUSH false
#define DEBUG false
#define TIME_CHECK true
#include <thread>
#include <future> 

#include "common/definitions.h"
#include "3rd_party/threadpool.h"
#include "optimizers/optimizers.h"
#include "training/training.h"
#include "training/validator.h"
#include "training/dropper.h"

namespace marian {

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

    virtual void load() = 0;

    virtual void save() = 0;
};


template <class Builder>
class AsyncGraphGroup : public GraphGroup {
  private:
    Ptr<Builder> builder_;

    std::vector<size_t> devices_;

    std::vector<Ptr<ExpressionGraph>> graphs_;

    std::mutex sync_;
    std::vector<std::mutex> shardSync_;

    std::vector<Tensor> params_;
    std::vector<Ptr<TensorAllocator> > paramsAlloc_;

    std::vector<Tensor> grads_;
    std::vector<SparseTensor> sparseGrads_;
    std::vector<SparseTensor> localSparseGrads_;

    std::vector<Ptr<TensorAllocator>> gradsAlloc_;

    std::vector<Ptr<OptimizerBase>> shardOpt_;

    std::vector<GradientDrop> gradDropper_;

    int shardSize_;

    ThreadPool pool_;
    long long el_1 = 0; //computation time
    long long el_2 = 0; // grad drop time
    long long el_3 = 0; // communication time

    void fetchParams(Tensor oldParams) {
      if(graphs_.size() < 2)
        return;

      // @TODO read guard on parameters
      int pos = 0;
      
      std::vector<std::thread> threads;
      for (int idx = 0; idx < devices_.size(); idx++) {
        threads.emplace_back( std::thread( [=](int idx, int pos) {
          //individual mutex per-shard
          std::lock_guard<std::mutex> guard( shardSync_[idx] );
          oldParams->subtensor(pos , params_[idx]->size())->copyFrom(params_[idx]);
        }, idx, pos) );

        pos += shardSize_;
      }
      for (auto &&t : threads) {
        t.join();
      }
    }


    void pushGradients(Tensor newGrads) {
      if(graphs_.size() < 2) {
        opt_->update(graphs_[0]);
      }
      else {
        // add instead of copy?
        std::vector<std::thread> threads;
        int pos = 0;
        for (int idx = 0; idx < devices_.size(); idx++) {
            threads.emplace_back( std::thread([=](int idx, int pos) {
            //individual mutex per-shard
            std::lock_guard<std::mutex> guard( shardSync_[idx] );
            auto start1 = std::chrono::system_clock::now();
            grads_[idx]->copyFrom(  newGrads->subtensor(pos , grads_[idx]->size() ) );
            auto end1 = std::chrono::system_clock::now();

            auto start2 = std::chrono::system_clock::now();
            shardOpt_[idx]->update(params_[idx], grads_[idx]);
            cudaStreamSynchronize(0);
            auto end2 = std::chrono::system_clock::now();

            if (idx == 0 && newGrads->getDevice() == 0){
              auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
              auto elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
              el_1 += elapsed2.count();
              el_3 += elapsed1.count();
            }

          } , idx, pos) );

          pos += shardSize_;
        }
        for(auto&& t : threads)
          t.join();
      }
    }

    void sparseFetchParams(Tensor oldParams) {
      if(graphs_.size() < 2)
        return;

      // @TODO read guard on parameters
      int pos = 0;
      
      std::vector<std::thread> threads;
      for (int idx = 0; idx < devices_.size(); idx++) {
        threads.emplace_back( std::thread( [=](int idx, int pos) {
          //individual mutex per-shard
          std::lock_guard<std::mutex> guard( shardSync_[idx] );


            cudaStreamSynchronize(0);
        }, idx, pos) );

        pos += shardSize_;
      }
      for (auto &&t : threads) {
        t.join();
      }
    }
    
    void sparsePushGradients(SparseTensor newGrads, Tensor oldParams, int wut) {
      if(graphs_.size() < 2) {
        opt_->update(graphs_[0]);
      }
      else {
        // add instead of copy?
        std::vector<std::thread> threads;
        
        int pos = 0;
        if (wut == 0)
          if (DEBUG) std::cerr<<"expected total size : "<<newGrads->size()<<std::endl;
        for (int idx = 0; idx < devices_.size(); idx++) {
          threads.emplace_back( std::thread([=](int idx, int pos, int wut) {
            //individual mutex per-shard
            //std::cerr<<"SPARSE PUSHING"std::endl;
            std::lock_guard<std::mutex> guard( shardSync_[idx] );
            auto start1 = std::chrono::system_clock::now();
            SparseTensor subGrad = newGrads->subtensor(pos , grads_[idx]->size() ,idx);
            sparseGrads_[idx]->copyFrom( subGrad  );
            auto end1 = std::chrono::system_clock::now();
            auto start2 = std::chrono::system_clock::now();
            //std::cout<<"subtensor size "<<sparseGrads_[idx]->size()<<std::endl;
            sparseGrads_[idx]->shiftIndices( -pos );
            sparseGrads_[idx]->toDense(grads_[idx], 0);
            shardOpt_[idx]->update(params_[idx], grads_[idx]);
            cudaStreamSynchronize(0);

            if (!SPARSE_PUSH)
              return;
              //now fetch
              sparseGrads_[idx]->scatterCopyFrom( params_[idx] );

              if (subGrad->size() != sparseGrads_[idx]->size()){
                if (DEBUG) std::cerr<<"LHO"<<std::endl;
                exit(1);
              }
              if (wut == 0){
                if (DEBUG) 
                  std::cerr<<"fraction "<<idx<<"   :  "<<subGrad->size()<<" / " <<subGrad->capacity()<<"    copying "<< oldParams->size() <<std::endl;
                //fprintf(stderr, "scatter updating %d\n", (int)oldParams->subtensor(pos, grads_[idx]->size())->size() );
              }

              auto end2 = std::chrono::system_clock::now();

              auto start3 = std::chrono::system_clock::now();
              subGrad->copyFrom(sparseGrads_[idx] , true);
              subGrad->scatterUpdate( oldParams->subtensor(pos , params_[idx]->size()) , -pos);
              
              cudaStreamSynchronize(0);
              auto end3 = std::chrono::system_clock::now();

            if (idx == 0 && newGrads->getDevice() == 0){
              auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
              auto elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
              auto elapsed3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
              el_1 += elapsed2.count();
              el_3 += elapsed1.count() + elapsed3.count();
            }
          } , idx, pos, wut) );

          pos += shardSize_;
        }
        for(auto&& t : threads)
          t.join();
      }
    }

      int bt = 0;
    void execute(Ptr<data::CorpusBatch> batch) {
      static bool first = true;
      if(first) {
        // initialize the parameters
        for(auto graph : graphs_) {
          builder_->build(graph, batch);
          graph->forward();
        }

        if(params_.size() == 0) {
          int totalSize = graphs_[0]->params().vals()->size();
          shardSize_ = ceil(totalSize / devices_.size());

          int pos = 0;
          //parameter sharding
          for (auto device : devices_){
            int __size__ = min(shardSize_, totalSize);
            totalSize -= __size__;
            Tensor param_;
            Ptr<TensorAllocator> allocator_ = New<TensorAllocator>(device);

            allocator_->reserveExact(__size__);
            allocator_->allocate(param_, {1, __size__});
            paramsAlloc_.push_back(allocator_);
            param_->copyFrom( graphs_[0]->params().vals()->subtensor( pos , __size__ ) );
            params_.push_back(param_);
            pos += __size__;

          }
        }
        if(grads_.size() == 0) {
          int totalSize = graphs_[0]->params().vals()->size();
          int sparseCap = totalSize / 10;
          for (auto device : devices_){
            int __size__ = min(shardSize_, totalSize);
            totalSize -= __size__;
            Tensor grad_;
            Ptr<TensorAllocator> allocator_ = New<TensorAllocator>(device);

            allocator_->reserveExact(__size__);
            allocator_->allocate(grad_, {1, __size__});
            gradsAlloc_.push_back(allocator_);
            grads_.push_back(grad_);
            //give size of 10% extra grads. even though we will use only around 1%
            sparseGrads_.push_back( SparseTensor(new SparseTensorBase( sparseCap, device )) );
            localSparseGrads_.push_back( SparseTensor(new SparseTensorBase(sparseCap , device )) );
          }
        } 

        first = false;
      } 

      auto task = [this](Ptr<data::CorpusBatch> batch) {
        static size_t i = 0;
        thread_local Ptr<ExpressionGraph> graph;
        thread_local size_t t = 0;
        thread_local size_t my_id = 0;

        if(!graph) {
          std::lock_guard<std::mutex> lock(sync_);
          my_id = i;
          graph = graphs_[i++];

          fetchParams(graph->params().vals());
          cudaStreamSynchronize(0);
        }
        auto start4 = std::chrono::system_clock::now();
        
        builder_->build(graph, batch);
        if (!GRAD_DROP || !SPARSE_PUSH)
          fetchParams(graph->params().vals());

        auto end4 = std::chrono::system_clock::now();

        auto start1 = std::chrono::system_clock::now();


        graph->forward();
        float cost = graph->topNode()->scalar();
        graph->backward();

        cudaStreamSynchronize(0);
        auto end1 = std::chrono::system_clock::now();
        auto start2 = std::chrono::system_clock::now();
        if (GRAD_DROP)
          gradDropper_[my_id].dropGraph(graph , localSparseGrads_[my_id] , 0.99 );
        cudaStreamSynchronize(0);

        auto end2 = std::chrono::system_clock::now();
        
        //if (GRAD_DROP)
        // sparsePushGradients(localSparseGrads_[my_id], graph->params().vals(), my_id );
        //else
          pushGradients(graph->params().grads());
        cudaStreamSynchronize(0);


        if(reporter_) {
          std::lock_guard<std::mutex> guard(sync_);
          reporter_->update(cost, batch);
          if(reporter_->batches % options_->get<size_t>("save-freq") == 0)
            this->save();
          reporter_->validate(graph);
        }

        if (TIME_CHECK && my_id == 0){
          bt++;
          auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
          auto elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
          auto elapsed4 = std::chrono::duration_cast<std::chrono::milliseconds>(end4 - start4);
          el_1 += elapsed1.count();
          el_2 += elapsed2.count();
          el_3 += elapsed4.count();
          if (bt == 25){
            std::cout<<"\nTime used per "<<25<<" batches\n";
            std::cout<<"    "<< el_1 <<"  COMP\n";
            std::cout<<"    "<< el_2 <<"  DROP\n";
            std::cout<<"    "<< el_3 <<"  DATA\n";
            bt = 0;
            el_1 = 0;
            el_2 = 0;
            el_3 = 0;
          }

        }

        t++;
      };

      pool_.enqueue(task, batch);
    }

    void load() {
      if(options_->has("init")) {
        std::string init = options_->get<std::string>("init");
        for(auto graph : graphs_)
          builder_->load(graph, init);
      }
    }

  public:
    typedef Builder builder_type;

    AsyncGraphGroup(Ptr<Config> options)
     : GraphGroup(options),
       builder_{New<Builder>(options_)},
       devices_{options_->get<std::vector<size_t>>("device")},
       pool_{devices_.size(), devices_.size()},
       shardSync_{devices_.size()},
       gradDropper_{devices_.size()} {

      for(auto device : devices_) {
        auto graph = New<ExpressionGraph>();
        graph->setDevice(device);
        graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
        graphs_.push_back(graph);
        shardOpt_.push_back(Optimizer(options_));
      }

      load();
    }

    void update(Ptr<data::CorpusBatch> batch) {
      execute(batch);
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

    void load() {
      if(options_->has("init")) {
        std::string init = options_->get<std::string>("init");
        for(auto graph : graphs_)
        builder_->load(graph, init);
      }
    }

  public:
    typedef Builder builder_type;

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

      load();
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
