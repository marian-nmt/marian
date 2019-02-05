#include "training/communicator.h"
#include "common/utils.h"

#if defined(CUDA_FOUND) && defined(USE_NCCL)
#include "training/communicator_nccl.h"
#endif

#if MPI_FOUND
#include "mpi.h"
#endif

namespace marian {

Ptr<ICommunicator> createCommunicator(
  const std::vector<Ptr<ExpressionGraph>>& graphs,
  bool noNccl, Ptr<IMPIWrapper> mpi) {
  mpi;
#if defined(CUDA_FOUND) && defined(USE_NCCL)
  if(noNccl) {
    LOG(warn, "[comm] NCCL communicator overridden");
    return New<DefaultCommunicator>(graphs, mpi);
  }

  // if at least one of the devices is not a gpu, fall-back to default
  for(auto& graph : graphs) {
    if(graph->getBackend()->getDeviceId().type == DeviceType::cpu) {
      return New<DefaultCommunicator>(graphs, mpi);
    }
  }

  size_t d = graphs.size();
  if((d & (d - 1)) != 0) {
    LOG(warn,
        "[comm] Number of devices {} is not a power of 2 and communication "
        "might be slow with NCCL",
        d);
    LOG(warn, "[comm] You can switch off NCCL with --no-nccl option", d);
  }

  // the actual implementation is inside communicator.cu
  return New<NCCLCommunicator>(graphs, mpi); 
#else // no CUDA or no NCCL
  noNccl; // (unused)
  return New<DefaultCommunicator>(graphs, mpi);
#endif
}

std::string IMPIWrapper::idStr() const { // helper to identify the node in logs
  std::string hostname; int pid; std::tie
  (hostname, pid) = utils::hostnameAndProcessId();
  return hostname + ":" + std::to_string(pid) + " MPI rank " + std::to_string(myMPIRank()) + " out of " + std::to_string(numMPIProcesses());
}

#if MPI_FOUND
// wrapper for MPI calls
// Since MPI can only be initialized once, only one instance of this class can exist.
class MPIWrapper : public IMPIWrapper
{
  int my_rank_;         // MPI rank of this node
  int comm_world_size_; // Number of nodes in MPI world (cluster)

  void handleError(int mpiRetval, const char* exprString) const { // call this with the return value of all MPI calls to report errors
    if (mpiRetval != MPI_SUCCESS) {
      char errStr[MPI_MAX_ERROR_STRING + 1] = { 0 };
      int resultLen = 0;
      MPI_Error_string(mpiRetval, &errStr[0], &resultLen);
      errStr[resultLen] = 0; // (@TODO: needed?)
      ABORT("MPI call failed with code {} '{}' on node {}: {}", mpiRetval, errStr, my_rank_, exprString); // @TODO: also log host name, which is involved on Windows
    }
  }
#define HANDLE_MPI_ERROR(expr) (handleError(expr, #expr)) // call through a macro so we can also log the failed expression itself

public:
  MPIWrapper(bool multiThreaded) {
    int requiredThreadingMode = multiThreaded ? MPI_THREAD_MULTIPLE : MPI_THREAD_FUNNELED; // FUNNELED means only one thread ever calls MPI

    int argc = 1; char* argv[] = { const_cast<char*>("this.exe") }; char** argvp = argv; // dummy argc/argv since MPI_Init needs something here
    int providedThreadingMode;
    HANDLE_MPI_ERROR(MPI_Init_thread(&argc, &argvp, MPI_THREAD_MULTIPLE, &providedThreadingMode));
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN); // have errors reported as return codes

    ABORT_IF(
      providedThreadingMode < requiredThreadingMode,
      "Your version of MPI does not support multi-threaded communication.");

    MPI_Comm_size(MPI_COMM_WORLD, &comm_world_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);

    // patch logging pattern to include the MPI rank, so that we can associate error messages with nodes
    if (numMPIProcesses() > 1) {
      std::string rankStr = std::to_string(MPIWrapper::myMPIRank());
      std::string maxRankStr = std::to_string(MPIWrapper::numMPIProcesses() -1);
      while (rankStr.size() < maxRankStr.size()) // pad so that logs across MPI processes line up nicely
        rankStr.insert(rankStr.begin(), ' ');
      switchtoMultinodeLogging(rankStr);
    }

    // log hostnames in order, and test
    for (size_t r = 0; r < numMPIProcesses(); r++) {
      MPIWrapper::barrier();
      if (r == MPIWrapper::myMPIRank() && MPIWrapper::numMPIProcesses() > 1) {
        std::string hostname; int pid; std::tie
        (hostname, pid) = utils::hostnameAndProcessId();
        LOG(info, "[mpi] Initialized as rank {} out of {} processes on {} as process {}",
                  MPIWrapper::myMPIRank(), MPIWrapper::numMPIProcesses(), hostname, pid);
      }
      MPIWrapper::barrier();
    }
  }

  virtual size_t myMPIRank()        const override { return (size_t)my_rank_; };
  virtual size_t numMPIProcesses() const override { return (size_t)comm_world_size_; };

  virtual void barrier(MPI_Comm comm = MPI_COMM_WORLD) const override {
    HANDLE_MPI_ERROR(MPI_Barrier(comm));
  }
  virtual void bCast(void* buf, size_t count, MPI_Datatype datatype, size_t rootRank, MPI_Comm comm = MPI_COMM_WORLD) const override {
    HANDLE_MPI_ERROR(MPI_Bcast(buf, (int)count, datatype, (int)rootRank, comm));
  }
  virtual void sSend(void* buf, size_t count, MPI_Datatype datatype, size_t destRank, int tag, MPI_Comm comm) const override {
    HANDLE_MPI_ERROR(MPI_Ssend(buf, (int)count, datatype, (int)destRank, tag, comm));
  }
  virtual void recv(void* buf, size_t count, MPI_Datatype datatype, size_t sourceRank, int tag, MPI_Comm comm, MPI_Status* status) const override {
    HANDLE_MPI_ERROR(MPI_Recv(buf, (int)count, datatype, (int)sourceRank, tag, comm, status));
  }
  virtual void allReduce(const void* sendbuf, void* recvbuf, size_t count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) const override {
    if (sendbuf == recvbuf)
      sendbuf = MPI_IN_PLACE; // MSMPI requires this
    HANDLE_MPI_ERROR(MPI_Allreduce(sendbuf, recvbuf, (int)count, datatype, op, comm));
  }
  virtual void finalize() override {
    HANDLE_MPI_ERROR(MPI_Finalize());
  }
};
#endif

// dummy MPI wrapper that implements only one process without actual operations
// This is used when not compiling under MPI.
class FakeMPIWrapper : public IMPIWrapper
{
public:
  FakeMPIWrapper(bool) {
    LOG(warn, "Compiled without MPI support. Falling back to FakeMPIWrapper");
  }

  virtual size_t myMPIRank() const override { return 0; };
  virtual size_t numMPIProcesses() const override { return 1; };

#pragma warning(push)
#pragma warning(disable: 4100) // unreferenced formal parameter
  // most functions are no-ops when applied to a single process
  virtual void barrier(MPI_Comm comm) const override {
    comm;
  }
  virtual void bCast(void* buf, size_t count, MPI_Datatype datatype, size_t rootRank, MPI_Comm comm) const override {
    buf; count; datatype; rootRank; comm;
  }
  virtual void sSend(void* buf, size_t count, MPI_Datatype datatype, size_t destRank, int tag, MPI_Comm comm) const override {
    buf; count; datatype; destRank; tag; comm;
  }
  virtual void recv(void* buf, size_t count, MPI_Datatype datatype, size_t sourceRank, int tag, MPI_Comm comm, MPI_Status* status) const override {
    buf; count; datatype; sourceRank; tag; comm;
    // @TODO: fill in status
    ABORT_IF(status != MPI_STATUS_IGNORE, "FakeMPIWrapper::recv() does not yet implement returning a status object");
  }
  virtual void allReduce(const void* sendbuf, void* recvbuf, size_t count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) const override {
    count; datatype; op; comm;
    // @TODO: There is only one place where this is called with sendbuf != recvbuf, which is sync multi-node.
    //        I think that can be changed to use the same buffer. Then we should change this API
    //        to only accept one parameter, and remove this error check can be removed.
    ABORT_IF(sendbuf != recvbuf, "FakeMPIWrapper::allReduce() only implemented for in-place operation"); // otherwise it's not a no-op, we must copy data
  }
#pragma warning(push)
  virtual void finalize() override { }
};

// create instance of the singleton MPI wrapper
static Ptr<IMPIWrapper> s_mpi;    // singleton instance of MPI wrapper
static size_t s_mpiUseCount;      // how many times has this wrapper been instantiated?
static bool s_mpiIsMultiThreaded; // multi-threading mode of this instance

Ptr<IMPIWrapper> initMPI(bool multiThreaded) {
  if (!s_mpi) {
#if MPI_FOUND
    s_mpi = New<MPIWrapper>(multiThreaded);
#else
    s_mpi = New<FakeMPIWrapper>(multiThreaded);
#endif
    s_mpiIsMultiThreaded = multiThreaded;
  }
  else {
    ABORT_IF(s_mpiIsMultiThreaded != multiThreaded, "attempted to reinitialize MPI with different multi-threading mode");
  }
  s_mpiUseCount++;
  return s_mpi;
}

void finalizeMPI(Ptr<IMPIWrapper>&& mpi) {
  ABORT_IF(mpi == nullptr || mpi != s_mpi, "attempted to finalize an inconsistent MPI instance. This should not be possible.");
  mpi = nullptr; // destruct caller's handle
  ABORT_IF(s_mpiUseCount == 0, "finalize called too many times. This should not be possible.");
  if (s_mpiUseCount == 1) { // last call finalizes MPI, i.e. tells MPI that we sucessfully completed computation
    ABORT_IF(s_mpi.use_count() != 1, "dangling reference to MPI??"); // caller kept another shared_ptr to this instance
    s_mpi->finalize(); // signal successful completion to MPI
    s_mpi = nullptr;   // release the singleton instance upon last finalization
  }
  s_mpiUseCount--;
}

}  // namespace marian
