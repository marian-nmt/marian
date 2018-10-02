// @TODO: rename to communicator_nccl.h
// Note: This must only be included if defined(CUDA_FOUND) && defined(USE_NCCL)
#include "training/communicator.h"
 
#include "cuda_runtime.h"
#include "nccl.h"
#include "tensors/gpu/cuda_helpers.h"


#include <signal.h> // HACK
#include <sys/types.h>
#include <sys/syscall.h>
pid_t gettid(void){ return syscall(SYS_gettid); }

namespace marian {

class NCCLCommunicator : public ICommunicator {
private:
  std::vector<ncclComm_t> comms_;     // [device index]
  std::vector<cudaStream_t> streams_; // [device index]
  std::vector<int> devices_;          // [device index]
  Ptr<IMPIWrapper> mpi_; // (may be null)

  void groupStart() const { NCCLCHECK(ncclGroupStart()); } // helpers to make sure we check the error
  //void groupEnd() const   { NCCLCHECK(ncclGroupEnd());   }
  void groupEnd() const {
    auto rc = ncclGroupEnd();
    if (rc != ncclSuccess)
      LOG(critical, "[{}] groupEnd failed", mpiIdStr());
    NCCLCHECK(rc);
  }

  void synchronizeAll() {
    for(int i = 0; i < graphs_.size(); ++i) {
      CUDA_CHECK(cudaSetDevice(devices_[i]));
      CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
    }
  }

  std::string mpiIdStr() const { // (for logging)
    return mpi_ ? mpi_->idStr() : "";
  }

  size_t myNcclRank(size_t localDeviceIndex) const { // map local device index to a global rank
    if (mpi_)
      return mpi_->myMPIRank() * devices_.size() + localDeviceIndex;
    else
      return localDeviceIndex;
  }

  size_t ncclRankToMPIRank(size_t ncclRank) const {
    if (mpi_)
      return ncclRank / devices_.size();
    else
      return ncclRank;
  }

  size_t ncclRankToLocalDeviceIndex(size_t ncclRank) const {
    return ncclRank % devices_.size();
  }

  size_t numNcclRanks() const { // total number of devices across all MPI processes
    if (mpi_)
      return mpi_->numMPIProcesses() * devices_.size();
    else
      return devices_.size();
  }

  size_t dataSize() const { // total number of floats that comprise the concatenated parameter and gradient vector
    return graphs_[0]->params()->vals()->size();
  }

  // determine the (max) shard size
  // All shards except the last one have this size.
  // Presently, even all shards must have identical size, due to a limitation in NCCL we have not yet worked around.
  size_t shardSize() const {
    size_t numShards = numNcclRanks();
    size_t size = (dataSize() + numShards - 1) / numShards;
#if 1 // for now, all shards must have the same size, since NCCL does not allow a sub-slice for the last shard
    ABORT_IF(size * numShards != dataSize(), "presently, all shards must have the same size");
#endif
    return size;
  }

  // determine the index range (begin, end) of a shard
  std::pair<size_t, size_t> ncclRankShardRange(size_t rank) const {
    size_t size = shardSize();
    size_t begin = rank * size;
    size_t end = begin + size;
    end = std::min(end, dataSize()); // clip last shard. Note: Presently this never happens, since shardSize() enforces uniform shard size.
    return{ begin, end };
  }

  // determine the index range (begin, end) of a shard
  std::pair<size_t, size_t> localShardRange(size_t localDeviceIndex) const {
    return ncclRankShardRange(myNcclRank(localDeviceIndex));
  }

  static std::string ncclVersionString() {
    int ncclVersion = 0;
    ncclGetVersion(&ncclVersion);
    return std::to_string(ncclVersion/1000) + "." + std::to_string((ncclVersion/100)%10) + "." + std::to_string(ncclVersion%100);
  }

  void mpiBarrier() const {
    if (mpi_)
      mpi_->barrier();
  }

public:
  // a NCCLCommunicator is bound to a set of graphs, one per GPU device
  // If MPI is used, then each MPI process has an instance of this class for its specific
  // set of GPU devices, which are communicating with each other. The total number of GPUs
  // involved in the NCCL communication setup is (#MPI processes) x (#GPUs per process).
  NCCLCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs, Ptr<IMPIWrapper> mpi)
      : ICommunicator(graphs),
        comms_(graphs.size()),
        streams_(graphs.size()),
        devices_(graphs.size()),
        mpi_(mpi) {
    mpiBarrier();
    LOG(info, "[comm] Using NCCL {} {}for GPU communication", ncclVersionString(), mpi_ ? "and MPI " : "");

    // set up our local devices
    for(int i = 0; i < graphs_.size(); ++i) {
      auto device = graphs_[i]->getBackend()->getDeviceId();

      ABORT_IF(device.type != DeviceType::gpu,
               "NCCL communicator can only be used with GPUs");

      devices_[i] = device.no;
      CUDA_CHECK(cudaSetDevice(devices_[i]));
      CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }

    // when using MPI, the setup is a laborious
    // cf. https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#multidevprothrd
    // generate NCCL unique ID at one process and broadcast to all
    ncclUniqueId uniqueId = { 0 };
    if (!mpi_ || mpi->myMPIRank() == 0)
      NCCLCHECK(ncclGetUniqueId(&uniqueId));

    if (mpi_) {
      //LOG(info, "[{}] before bcast", mpiIdStr());
      static_assert(sizeof(uniqueId) == NCCL_UNIQUE_ID_BYTES, "wrong NCCL_UNIQUE_ID_BYTES??"); // (this value is used in NVidia examples)
      mpi_->bCast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0);
      //LOG(info, "[{}] after bcast", mpiIdStr());
    }

      mpiBarrier();
#define SIG_BAD 27 // SIGPROF
    //LOG(info, "[{}] setting fake handlers", mpiIdStr());
    //// FAKE SIGNAL HANDLERS
    //size_t sig = SIG_BAD;//for (size_t sig = 0; sig < NSIG; sig++) {
    //  struct sigaction sa = { 0 };
    //  sigemptyset(&sa.sa_mask);
    //  sa.sa_flags = SA_RESTART;
    //  sa.sa_handler = [&](int signal){
    //    char hostnamebuf[HOST_NAME_MAX + 1] = { 0 };
    //    gethostname(hostnamebuf, sizeof(hostnamebuf));
    //    LOG(info, "[{}:{}:{}] Signal {} caught--still??", hostnamebuf, getpid(), (int)gettid(), signal);
    //  };
    //  auto rc1 =
    //  sigaction(sig, &sa, nullptr);
    //  LOG(info, "[{}] {} -> {}", mpiIdStr(), sig, rc1);
    //}
    //LOG(info, "[{}] done setting fake handlers", mpiIdStr());

      sigset_t newSigSet, oldSigSet;

      pthread_sigmask(SIG_BLOCK, NULL, &newSigSet);
      LOG(info, "[{}] pthread_sigmask original mask={}", mpiIdStr(), newSigSet.__val[0]);

      sigemptyset(&newSigSet);
      sigaddset(&newSigSet, SIG_BAD);
      LOG(info, "[{}] pthread_sigmask mask={}", mpiIdStr(), newSigSet.__val[0]);
      auto rc2 = pthread_sigmask(SIG_BLOCK, &newSigSet, &oldSigSet);
      LOG(info, "[{}] pthread_sigmask rc={}", mpiIdStr(), rc2);
      auto rc2b = sigprocmask(SIG_BLOCK, &newSigSet, &oldSigSet);
      LOG(info, "[{}] pthread_sigmask rc={}", mpiIdStr(), rc2b);

      pthread_sigmask(SIG_BLOCK, NULL, &newSigSet);
      LOG(info, "[{}] pthread_sigmask resulting mask={}", mpiIdStr(), newSigSet.__val[0]);
      sigprocmask(SIG_BLOCK, NULL, &newSigSet);
      LOG(info, "[{}:{}] sigprocmask resulting mask={}", mpiIdStr(), (int)gettid(), newSigSet.__val[0]);

    // @BUGBUG: This fails randomly for 4, 32, and 64 GPUs. Seems stable for 8 and 16?
    //          Failed, NCCL error 2 'unhandled system error' - ncclGroupEnd()
    //          include/shm.h:26 NCCL WARN Unable to allocate shared memory (4263936 bytes) : Interrupted system call
      // if more than one device then initialize NCCL with group API
      //if (devices_.size() > 1) {
    for (size_t attempts = 1; ; attempts++) {
      //mpiBarrier();
      LOG(info, "[{}] groupStart", mpiIdStr());
      groupStart();
      for (int localDeviceIndex = 0; localDeviceIndex < devices_.size(); localDeviceIndex++) {
        CUDA_CHECK(cudaSetDevice(devices_[localDeviceIndex]));
        LOG(info, "[{}] ncclCommInitRank {} out of {}: GPU[{}]", mpiIdStr(), myNcclRank(localDeviceIndex), numNcclRanks(), localDeviceIndex);
        NCCLCHECK(ncclCommInitRank(&comms_[localDeviceIndex], numNcclRanks(), uniqueId, myNcclRank(localDeviceIndex)));
        //LOG(info, "[{}] done ncclCommInitRank {} out of {}, GPU[{}]", mpiIdStr(), myNcclRank(localDeviceIndex), numNcclRanks(), localDeviceIndex);
      }
      //groupEnd();
      //LOG(info, "[{}] barrier before groupEnd", mpiIdStr());
      //mpiBarrier();
      //::sleep(1); // where does this signal come from? Maybe waiting helps

      LOG(info, "[{}] groupEnd", mpiIdStr());
      auto rc = ncclGroupEnd(); // things happen here

      LOG(info, "[{}] groupEnd rc = {}", mpiIdStr(), rc);
      ::sleep(/*seconds=*/mpi_ ? mpi_->numMPIProcesses() : 0); // give all a chance to detect their error. This makes a difference.
      int err = (rc != ncclSuccess);
      int totalErr = err;
      // determine the error situation across all workers
      if (mpi_) {
        LOG(info, "[{}] barrier before allReduce err", mpiIdStr());
        mpiBarrier();
        LOG(info, "[{}] allReduce err {}", mpiIdStr(), err);
        mpi_->allReduce(&err, &totalErr, 1, MPI_INT, MPI_SUM);
        LOG(info, "[{}] groupEnd failure {} total failures {}", mpiIdStr(), err, totalErr);
      }
      if (totalErr == 0)
        break;
      LOG(info, "[{}] groupEnd failed: {}", mpiIdStr(), rc);
      ::sleep(1); // wait some more before taking the process down
      ABORT_IF(attempts == 5, "too many failed retries");
      if (!err) // (does not seem to happen, since all fail)
        for (int localDeviceIndex = 0; localDeviceIndex < devices_.size(); localDeviceIndex++) {
          LOG(info, "[{}] cleaning up {}: rc was {}", mpiIdStr(), localDeviceIndex, rc);
          ncclCommDestroy(comms_[localDeviceIndex]);
        }
      mpiBarrier();
      LOG(info, "[{}] sleeping", mpiIdStr());
      ::sleep(5); // sleeping
      LOG(info, "[{}] attempt {}", mpiIdStr(), attempts+1);
    }

      auto rc3 = pthread_sigmask(SIG_SETMASK, &oldSigSet, NULL);
      LOG(info, "[{}] pthread_sigmask reset rc={}", mpiIdStr(), rc3);
    LOG(info, "[{}] groupEnd succeeded", mpiIdStr());
    mpiBarrier();
      //}
      //// one device: no group API
      //else {
      //  CUDA_CHECK(cudaSetDevice(devices_[0]));
      //  LOG(info, "[mpi rank {} of {}] ncclCommInitRank", mpi_->myMPIRank(), mpi_->numMPIProcesses());
      //  NCCLCHECK(ncclCommInitRank(&comms_[0], mpi_->numMPIProcesses(), uniqueId, mpi_->myMPIRank()));
      //  LOG(info, "[mpi rank {}] done constructing NCCLCommunicator", mpi_->myMPIRank());
      //}
    //}
    //// without MPI, we have a handy convenience version to initialize
    //// @TODO: We should be able to just use the code above as well.
    //else {
    //  LOG(info, "ncclCommInitAll");
    //  NCCLCHECK(ncclCommInitAll(comms_.data(), devices_.size(), devices_.data()));
    //  LOG(info, "done ncclCommInitAll");
    //}
    LOG(info, "[{}] NCCLCommunicator constructed successfully", mpiIdStr());
  }

  ~NCCLCommunicator() override {
    for(int i = 0; i < devices_.size(); ++i) {
      cudaSetDevice(devices_[i]);
      cudaStreamDestroy(streams_[i]);
      ncclCommDestroy(comms_[i]);
    }
  }

  void foreach(const ForeachFunc& func, bool parallel = true, bool localShardsOnly = true) const override {
    parallel &= graphs_.size() > 1;

    // This loop is dual-purpose:
    //  - localShardsOnly=true:  iterate over all shards on *this* MPI process
    //  - localShardsOnly=false: iterate over all shards on the entire NCCL setup
    // These differ in multi-MPI-process configurations.
    std::vector<std::thread> group;
    for(size_t i = 0; i < localShardsOnly ? graphs_.size() : numNcclRanks(); ++i) {
      size_t begin, end; std::tie
      (begin, end) = localShardsOnly ? localShardRange(i) : ncclRankShardRange(i);
      //std::cerr << "[" << mpiIdStr() << "] foreach " << begin << " " << end << std::endl;
      size_t size = end-begin;

      if (parallel)
        group.emplace_back(func, i, begin, end);
      else
        func(i, begin, end);
    }
    for(auto& t : group) // (note: group is empty is not parallel)
      t.join();
  }

  void scatterReduce() override {
    groupStart();
    for(int i = 0; i < graphs_.size(); ++i) {
      size_t begin, end; std::tie
      (begin, end) = localShardRange(i);
      //std::cerr << "[" << mpiIdStr() << "] scatterReduce " << begin << " " << end << std::endl;

      auto grads = graphs_[i]->params()->grads();
      const auto* sendbuf = grads->data();
      auto*       recvbuf = grads->subtensor(begin, end-begin)->data();
      size_t      bufsize = shardSize();

      NCCLCHECK(ncclReduceScatter(sendbuf, recvbuf, bufsize, ncclFloat, ncclSum, comms_[i], streams_[i]));
    }
    groupEnd();
    //std::cerr << "scatterReduce submitted" << std::endl;
    synchronizeAll();
    //std::cerr << "scatterReduce completed" << std::endl;
  }

  void allGather() override {
    groupStart();
    for(int i = 0; i < graphs_.size(); ++i) {
      size_t begin, end; std::tie
      (begin, end) = localShardRange(i);
      //std::cerr << "[" << mpiIdStr() << "] allGather " << begin << " " << end << std::endl;

      auto vals = graphs_[i]->params()->vals();
      const auto* sendbuf = vals->subtensor(begin, end-begin)->data();
      void*       recvbuf = vals->data();
      size_t      bufsize = shardSize();

      NCCLCHECK(ncclAllGather(sendbuf, recvbuf, bufsize, ncclFloat, comms_[i], streams_[i]));
    }
    groupEnd();
    synchronizeAll();
  }

  // swap distributed paramShards with model params()
  // It is assumed that all model params() on all devices and MPI processes are identical.
  // This is used for the smoothed parameters, and also for persisting optimizer state.
  void swapParams(const std::vector<Tensor>& distributedParamShards) override {
    ABORT_IF(mpi_ && mpi_->numMPIProcesses() > 1, "swapParams() support for MPI is not yet implemented");

    // get everything onto the CPU
    auto distributedParams = gatherState([&](size_t localDeviceIndex) {
      std::vector<float> tmp;
      distributedParamShards[localDeviceIndex]->get(tmp);
      return tmp;
    });
    // Now all MPI processes hold an identical copy of a concatenation of all distributedParamShards[] across local and remote devices.
    std::vector<float> localParams;
    graphs_[0]->params()->vals()->get(localParams);
    // Now all MPI processes hold an identical copy of params() (remember, we assumed all devices hold the same params()).
    ABORT_IF(distributedParams.size() != localParams.size(), "distributed sharded and local params have different size??");

    // swap
    std::swap(distributedParams, localParams);

    // distribute it back
    scatterState(distributedParams, [&](size_t localDeviceIndex,
                                        std::vector<float>::const_iterator begin,
                                        std::vector<float>::const_iterator end){
      std::vector<float> tmp(begin, end);
      distributedParamShards[localDeviceIndex]->set(tmp);
    });
    for (auto& graph : graphs_) // broadcast to every local graph
      graph->params()->vals()->set(localParams);

#if 0
    // Update all graphs with parameter shard
    // This function is called for each shard of MPI process[0] params().
    auto gather = [this, distributedParamShards](size_t ncclRank, size_t begin, size_t end) {
      // copy parameter shard to each graph, apart from last graph
      for(int i = 0; i < graphs_.size() - 1; ++i) {
        auto subParam
            = graphs_[i]->params()->vals()->subtensor(begin, end-begin);
        subParam->copyFrom(distributedParamShards[ncclRank]);
      }

      // back-up shard from last graph
      auto subParamLast
          = graphs_.back()->params()->vals()->subtensor(begin, end-begin);
      distributedParamShards[ncclRank]->copyFrom(subParamLast);

      auto subParamFirst
          = graphs_.front()->params()->vals()->subtensor(begin, end-begin);
      subParamLast->copyFrom(subParamFirst);
    };

    // execute for each shard across the entire NCCL configuration
    foreach(gather, /*parallel=*/false, /*localShardsOnly=*/false);
#endif
  }

  // Distribute a single CPU-side vector to shards across multiple devices and MPI processes.
  // This is used when restoring optimizer state, which is sharded.
  void scatterState(const std::vector<float>& data, const OptimizerBase::ScatterStateSetFunc& setFn) const override {
    ABORT_IF(mpi_ && mpi_->numMPIProcesses() > 1, "scatterState() support for MPI is not yet implemented");
    size_t dataSize = data.size();
    size_t numLocalDevices = graphs_.size();
    size_t shardSize = (dataSize + numLocalDevices - 1) / numLocalDevices;// (size_t)(ceil(dataSize / (float)numLocalDevices));
    for(size_t localDeviceIndex = 0; localDeviceIndex < numLocalDevices; localDeviceIndex++) {
      size_t begin = localDeviceIndex * shardSize;
      size_t end   = std::min(begin + shardSize, dataSize);
      setFn(localDeviceIndex, data.begin() + begin, data.begin() + end);
    }
  }

  // Collect shards across multiple devices and MPI processes in the NCCL configuration into a single CPU-side vector.
  // This is used when persisting optimizer state, which is sharded.
  std::vector<float> gatherState(const OptimizerBase::GatherStateGetFunc& getFn) const override {
    ABORT_IF(mpi_ && mpi_->numMPIProcesses() > 1, "gatherState() support for MPI is not yet implemented");
    std::vector<float> data; // we know the size here
    for (size_t localDeviceIndex = 0; localDeviceIndex < graphs_.size(); localDeviceIndex++) {
      std::vector<float> tmp = getFn(localDeviceIndex);
      data.insert(data.end(), tmp.begin(), tmp.end());
    }
    ABORT_IF(data.size() != graphs_[0]->params()->vals()->size(), "gathering wrong amount of data??");
    return data;
  }

#if 0
  void pushParams(std::vector<Tensor>& paramShards) override {
    ABORT_IF(mpi_, "pushParams() support for MPI is not yet implemented");
    // Copy paramter shard from i-th graph to shard paramShards[i].
    // Graphs and shards with the same index live on the same device.

    auto copy = [this, paramShards](size_t idx, size_t begin, size_t end) {
      // copy parameter shard to each graph
      auto subParam
          = graphs_[idx]->params()->vals()->subtensor(begin, paramShards[idx]->size());
      paramShards[idx]->copyFrom(subParam);
    };

    foreach(copy);
  }

  void pullParams(const std::vector<Tensor>& paramShards) override {
    ABORT_IF(mpi_, "pullParams() support for MPI is not yet implemented");
    // Update all graphs with parameter shard

    auto gather = [this, paramShards](size_t idx, size_t begin, size_t end) {
      // copy parameter shard to each graph
      for(auto graph : graphs_) {
        auto subParam
            = graph->params()->vals()->subtensor(begin, paramShards[idx]->size());
        subParam->copyFrom(paramShards[idx]);
      }
    };
    foreach(gather);
  }
#endif

  // Doesn't work yet with NCCL
  // void pushParams(std::vector<Tensor>& params) {
  //   // Copy paramter shard from i-th graph to shard params[i].
  //   // Graphs and shards with the same index live on the same device.

  //   int pos = 0;
  //   for(int i = 0; i < graphs_.size(); ++i) {
  //     auto subParam = graphs_[i]->params()->vals()->subtensor(pos,
  //                                                             params[i]->size());
  //     groupStart();
  //     ncclBroadcast(subParam->data(),
  //                   params[i]->data(),
  //                   params[i]->size(),
  //                   ncclFloat,
  //                   0,
  //                   comms_[i],
  //                   streams_[i]);
  //     groupEnd();
  //     pos += params[i]->size();
  //   }
  //   synchronizeAll();
  // }

  // void pullParams(const std::vector<Tensor>& params) {
  //   // Update all graphs with parameter shard

  //   int totalSize = graphs_[0]->params()->vals()->size();
  //   int shardSize = ceil(totalSize / (float)graphs_.size());

  //   groupStart();
  //   for(int i = 0; i < graphs_.size(); ++i) {

  //     const void* sendbuff = (const void*)params[i]->data();
  //     void* recvbuff = (void*)graphs_[i]->params()->vals()->data();

  //     ncclAllGather(sendbuff,
  //                   recvbuff,
  //                   shardSize,
  //                   ncclFloat,
  //                   comms_[i],
  //                   streams_[i]);
  //   }
  //   groupEnd();

  //   synchronizeAll();
  // }
};

}  // namespace marian
