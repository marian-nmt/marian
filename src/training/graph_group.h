#pragma once

#include <thread>
#include <future>
#include <boost/filesystem.hpp>

#include "common/definitions.h"
#include "3rd_party/threadpool.h"
#include "optimizers/optimizers.h"
#include "training/training.h"
#include "training/validator.h"

#include "training/dropper.h"

#define HISTORY_SIZE 6
#define DROP_SIZE 0.999

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
    std::vector<Ptr<Builder>> builders_;

    std::vector<size_t> devices_;

    std::vector<Ptr<ExpressionGraph>> graphs_;

    std::mutex sync_;
    std::vector<std::mutex> shardSync_;

    std::vector<Tensor> params_[HISTORY_SIZE];
    std::vector<Tensor> tmpTensor, tmpDelta;


    std::vector<Tensor> grads_;
    std::vector<Ptr<TensorAllocator>> allocators;

    std::vector<Ptr<OptimizerBase>> shardOpt_;

    std::vector<GradientDrop> gradDropper_;
    std::vector<std::vector<GradientDrop>> fetchDropper_;


    std::vector<SparseTensor> localSparseGrads_;
    std::vector<SparseTensor> sparseGrads_;
    std::vector<SparseTensor> tmpSparseDelta;
    std::vector<std::vector<SparseTensor>> localSparseDelta;

    std::vector<int> globalVersionNumber; //version number per-shard

    std::vector<std::vector<int>> localVersionNumbers; //each worker has the version number obtained from each shard



    int shardSize_;

    ThreadPool pool_;

    void sparseFetchParams(Tensor oldParams, int worker_id ) {
      if(graphs_.size() < 2)
        return;

      // @TODO read guard on parameters
      int p = 0;

      std::vector<std::thread> threads;
      for (int i = 0; i < devices_.size(); i++) {
        threads.emplace_back( std::thread( [=](int idx, int pos) {
          //individual mutex per-shard
          std::lock_guard<std::mutex> guard( shardSync_[idx] );
          //obtain the delta
          int latestVersion =  globalVersionNumber[idx] % HISTORY_SIZE;
          int currVersion = localVersionNumbers[worker_id][idx] % HISTORY_SIZE;
          //check if the current version is too old
          if (globalVersionNumber[idx] - localVersionNumbers[worker_id][idx] >= HISTORY_SIZE)
            currVersion = (1 + globalVersionNumber[idx]) % HISTORY_SIZE; //if so, pick the best you can do

          //if already latest
          if (globalVersionNumber[idx] == localVersionNumbers[worker_id][idx])
            return;
          
//          printf("DOING THINGS ON GPU %d -> %d %d | %d %d %d\n", tmpTensor[idx]->getDevice(),  params_[latestVersion][idx]->getDevice(), tmpSparseDelta[idx]->getDevice(), 
//                                                                localSparseDelta[worker_id][idx]->getDevice(), tmpDelta[worker_id]->getDevice(), oldParams->getDevice());

          //update params. add with delta of latest param and current param
          //get delta
          Element(_1 = _2 - _3, tmpTensor[idx], params_[latestVersion][idx] , params_[currVersion][idx]);
          cudaStreamSynchronize(0);   
          //get sparse delta  
          fetchDropper_[worker_id][idx]->dropGraph(tmpTensor[idx] , tmpSparseDelta[idx] , DROP_SIZE );

          cudaStreamSynchronize(0);
          //move sparse delta
          localSparseDelta[worker_id][idx]->copyFrom( tmpSparseDelta[idx] );
          cudaStreamSynchronize(0);
          //obtain the delta
          localSparseDelta[worker_id][idx]->toDense( tmpDelta[worker_id]->subtensor(pos , grads_[idx]->size()) , 0 );
          cudaStreamSynchronize(0);
          //apply

          Element(_1 += _2 , oldParams->subtensor(pos , grads_[idx]->size()) , 
                             tmpDelta[worker_id]->subtensor(pos , grads_[idx]->size()));
          cudaStreamSynchronize(0);
          localVersionNumbers[worker_id][idx] =  globalVersionNumber[idx];

        }, i, p) );

        p += shardSize_;
      }
      for (auto &&t : threads) {
        t.join();
      }
    }

    void fetchParams(Tensor oldParams, int worker_id ) {
      if(graphs_.size() < 2)
        return;

      // @TODO read guard on parameters
      int p = 0;

      std::vector<std::thread> threads;
      for (int i = 0; i < devices_.size(); i++) {
        threads.emplace_back( std::thread( [=](int idx, int pos) {
          //individual mutex per-shard
          std::lock_guard<std::mutex> guard( shardSync_[idx] );
          //obtain the delta
          int latestVersion =  globalVersionNumber[idx] % HISTORY_SIZE;
          int currVersion = localVersionNumbers[worker_id][idx] % HISTORY_SIZE;
          //check if the current version is too old
          if (globalVersionNumber[idx] - localVersionNumbers[worker_id][idx] >= HISTORY_SIZE)
            currVersion = (1 + globalVersionNumber[idx]) % HISTORY_SIZE; //if so, pick the best you can do

          //if already latest
          if (globalVersionNumber[idx] == localVersionNumbers[worker_id][idx])
            return;
          
          //printf("DOING THINGS ON GPU %d %d %d %d\n", tmpTensor[idx]->getDevice(),  params_[latestVersion][idx]->getDevice(), tmpDelta[worker_id]->getDevice(), oldParams->getDevice());

          //update params. add with delta of latest param and current param
          //get delta
          Element(_1 = _2 - _3, tmpTensor[idx], params_[latestVersion][idx] , params_[currVersion][idx]);

          
          //move delta
          tmpDelta[worker_id]->subtensor(pos , grads_[idx]->size())->copyFrom(tmpTensor[idx]);

          //apply
          Element(_1 += _2 , oldParams->subtensor(pos , grads_[idx]->size()) , 
                             tmpDelta[worker_id]->subtensor(pos , grads_[idx]->size()));
          
          localVersionNumbers[worker_id][idx] =  globalVersionNumber[idx];

        }, i, p) );

        p += shardSize_;
      }
      for (auto &&t : threads) {
        t.join();
      }
    }

    void initFetchParams(Tensor oldParams, int worker_id ) {
      if(graphs_.size() < 2)
        return;

      // @TODO read guard on parameters
      int pos = 0;

      std::vector<std::thread> threads;
      for (int idx = 0; idx < devices_.size(); idx++) {
        threads.emplace_back( std::thread( [=](int idx, int pos) {
          //individual mutex per-shard
          std::lock_guard<std::mutex> guard( shardSync_[idx] );
          //obtain everything
          int latestVersion = globalVersionNumber[idx] % HISTORY_SIZE;
          oldParams->subtensor(pos , grads_[idx]->size())->copyFrom(params_[latestVersion][idx]);
        }, idx, pos) );

        pos += shardSize_;
      }
      for (auto &&t : threads) {
        t.join();
      }
    }

    void pushGradients(Tensor newGrads, int worker_id ) {
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
            grads_[idx]->copyFrom( newGrads->subtensor(pos , grads_[idx]->size() ) );

            // apply and increment your version number
            int pastVersion = globalVersionNumber[idx] % HISTORY_SIZE;
            int latestVersion =  ++globalVersionNumber[idx] % HISTORY_SIZE;
            params_[latestVersion][idx]->copyFrom(params_[pastVersion][idx]);
            shardOpt_[idx]->update(params_[ latestVersion ][idx],  grads_[idx] );

            cudaStreamSynchronize(0);
          } , idx, pos) );

          pos += shardSize_;
        }
        for(auto&& t : threads)
          t.join();
      }
    }

    void sparsePush(SparseTensor newGrads, int worker_id ) {
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
            // split to shard
            SparseTensor subGrad = newGrads->subtensor(pos , grads_[idx]->size() ,idx);

            cudaStreamSynchronize(0);
            // sent
            sparseGrads_[idx]->copyFrom(subGrad);
      
            cudaStreamSynchronize(0);      
            //convert back to dense, with index offset of -pos
            sparseGrads_[idx]->toDense(grads_[idx], -pos);

            cudaStreamSynchronize(0);
            
            // apply and increment your version number
            int pastVersion = globalVersionNumber[idx] % HISTORY_SIZE;
            int latestVersion =  ++globalVersionNumber[idx] % HISTORY_SIZE;
            params_[latestVersion][idx]->copyFrom(params_[pastVersion][idx]);
            shardOpt_[idx]->update(params_[ latestVersion ][idx],  grads_[idx] );

            cudaStreamSynchronize(0);
          } , idx, pos) );

          pos += shardSize_;
        }
        for(auto&& t : threads)
          t.join();
      }
    }

    Tensor newTensor(int size, int device){
      Tensor T;
      Ptr<TensorAllocator> allocator_ = New<TensorAllocator>(device);
      allocator_->reserveExact(size);
      allocator_->allocate(T, {1, size});
      allocators.push_back(allocator_);

      return T;
    }

    void execute(Ptr<data::CorpusBatch> batch) {
      static bool first = true;
      if(first && graphs_.size() > 1) {
        // initialize the parameters
        for(size_t i = 0; i < graphs_.size(); ++i) {
          builders_[i]->build(graphs_[i], batch);
          graphs_[i]->forward();
          globalVersionNumber.push_back(0);
          std::vector<int> localVersion;
          for (int j=0;j<graphs_.size();j++)
            localVersion.push_back(0);

          localVersionNumbers.push_back(localVersion);
        }

        if(params_[0].size() == 0) {
          int totalSize = graphs_[0]->params().vals()->size();
          shardSize_ = ceil(totalSize / devices_.size());

          int pos = 0;
          //parameter sharding
          for (auto device : devices_){
            int __size__ = min(shardSize_, totalSize);
            totalSize -= __size__;
            
            
            for (int h_id = 0; h_id < HISTORY_SIZE; h_id++){
              Tensor param_ = newTensor(__size__, device);
              param_->copyFrom( graphs_[0]->params().vals()->subtensor( pos , __size__ ) );
              params_[h_id].push_back(param_);
            }

            tmpTensor.push_back( newTensor(__size__, device) );

            pos += __size__;

          }
        }
        
        if(grads_.size() == 0) {
          int totalSize = graphs_[0]->params().vals()->size();
          int sparseCap = totalSize / 20;
          for (auto device : devices_){
            int __size__ = min(shardSize_, totalSize);
            totalSize -= __size__;

            grads_.push_back( newTensor( __size__, device  ) );

            //give size of 10% extra grads. even though we will use only around 1%
            sparseGrads_.push_back( SparseTensor(new SparseTensorBase( sparseCap, device )) );
            localSparseGrads_.push_back( SparseTensor(new SparseTensorBase(sparseCap , device )) );
            tmpSparseDelta.push_back( SparseTensor(new SparseTensorBase(sparseCap / devices_.size() , device )) );
            std::vector<SparseTensor> tmp;
            for (int i=0;i<devices_.size();i++)
              tmp.push_back( SparseTensor(new SparseTensorBase(sparseCap / devices_.size() , device )) );
            localSparseDelta.push_back(tmp);
          }
        }

        first = false;
      }

      auto task = [this](Ptr<data::CorpusBatch> batch) {
        static size_t i = 0;
        thread_local Ptr<ExpressionGraph> graph;
        thread_local Ptr<Builder> builder;
        thread_local size_t t = 0;

        thread_local size_t my_id = 0;


        if(!graph) {
          std::lock_guard<std::mutex> lock(sync_);
          graph = graphs_[i];
          my_id = i;
          builder = builders_[i++];

          tmpDelta.push_back( newTensor( graph->params().vals()->size() , graph->params().vals()->getDevice() ) );
        }

        builder->build(graph, batch);
        if (!t)
          initFetchParams(graph->params().vals() , my_id );
        else
          //fetchParams(graph->params().vals() , my_id );
          sparseFetchParams(graph->params().vals() , my_id );

        graph->forward();
        float cost = graph->topNode()->scalar();
        graph->backward();
  
        cudaStreamSynchronize(0);
        
        gradDropper_[my_id]->dropGraph(graph , localSparseGrads_[my_id] , DROP_SIZE );
        //cudaStreamSynchronize(0);

        //sparsePush
        sparsePush(localSparseGrads_[my_id], my_id);
        //pushGradients(graph->params().grads(), my_id);

        if(reporter_) {
          std::lock_guard<std::mutex> guard(sync_);
          reporter_->update(cost, batch);
          if(reporter_->batches % options_->get<size_t>("save-freq") == 0)
            this->save();
          size_t prevStalled = reporter_->stalled();
          reporter_->validate(graph);
          if(prevStalled < reporter_->stalled())
            for(auto opt : shardOpt_)
              opt->updateSchedule();
        }

        t++;
      };

      pool_.enqueue(task, batch);
    }

  public:
    typedef Builder builder_type;

    AsyncGraphGroup(Ptr<Config> options)
     : GraphGroup(options),
       devices_{options_->get<std::vector<size_t>>("devices")},
       pool_{devices_.size(), devices_.size()},
       shardSync_{devices_.size()} {

      for(auto device : devices_) {
        auto graph = New<ExpressionGraph>();
        graph->setDevice(device);
        graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
        graphs_.push_back(graph);
        shardOpt_.push_back(Optimizer(options_));
        gradDropper_.push_back(GradientDrop(new GradientDropBase()));
        std::vector<GradientDrop> tmp;
        for (int i=0;i<devices_.size();i++)
          tmp.push_back(GradientDrop(new GradientDropBase()));
        fetchDropper_.push_back(tmp);
        builders_.push_back(New<Builder>(options_));
      }
    }

    void update(Ptr<data::CorpusBatch> batch) {
      execute(batch);
    }

    void load() {
      if(!options_->get<bool>("no-reload")) {
        std::string init = options_->get<std::string>("model");
        if(boost::filesystem::exists(init)) {
          size_t i = 0;
          reporter_->load(init);
          for(auto graph : graphs_)
            builders_[i++]->load(graph, init);
        }
      }
    }

    void save() {
      if(options_->get<bool>("overwrite")) {
        std::string name = options_->get<std::string>("model");
        builders_[0]->save(graphs_[0], name, true);
        reporter_->save(name);
      }
      else {
        std::string name = options_->get<std::string>("model");
        std::string nameOverwrite = name;
        nameOverwrite.replace(name.size() - 4, 4,
          ".iter" + std::to_string(reporter_->batches) + ".npz");
        builders_[0]->save(graphs_[0], nameOverwrite);

        builders_[0]->save(graphs_[0], name, true);
        reporter_->save(name);
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

      auto devices = options_->get<std::vector<size_t>>("devices");
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
