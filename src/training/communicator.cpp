#include "training/communicator.h"
#if MPI_FOUND
#include "mpi.h"
#endif

namespace marian {

// Compile this if cuda is not being compiled.
// Version with CUDA and/or NCCL is compiled in communicator.cu
#ifndef CUDA_FOUND
Ptr<Communicator> createCommunicator(
    const std::vector<Ptr<ExpressionGraph>>& graphs,
    bool /*noNccl*/) {
  return New<DefaultCommunicator>(graphs);
}
#endif

#if MPI_FOUND
class MPIWrapper : public IMPIWrapper
{
  static std::mutex s_mutex;    // guards the following static variables
  static size_t s_useCount;     // how many times has this wrapper been instantiated?
  static int s_threadingMode;   // current level of thread support
  static int s_my_rank;         // MPI rank of this node
  static int s_comm_world_size; // Number of nodes in MPI world (cluster)

  static void handleError(int mpiRetval, const char* exprString) { // call this with the return value of all MPI calls to report errors
    if (mpiRetval != MPI_SUCCESS) {
      ABORT("MPI call failed with code {} on node {}: {}", mpiRetval, s_my_rank, exprString); // @TODO: also log host name, which is involved on Windows
    }
  }
#define HANDLE_MPI_ERROR(expr) (handleError(expr, #expr)) // call through a macro so we can also log the failed expression itself

public:
  MPIWrapper(bool sync) {
    int required_mode = sync ? MPI_THREAD_SERIALIZED : MPI_THREAD_MULTIPLE;

    std::lock_guard<std::mutex> guard(s_mutex);

    if (s_useCount == 0) {
      int argc = 1; char* argv[] = { const_cast<char*>("this.exe") }; char** argvp = argv; // dummy argc/argv since MPI_Init needs something here
      HANDLE_MPI_ERROR(MPI_Init_thread(&argc, &argvp, MPI_THREAD_MULTIPLE, &s_threadingMode)); // @TODO: should set_errhandler() be done before this?
      s_useCount++;
      MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN); // have errors reported as return codes

      if (s_threadingMode < required_mode)
        LOG(warn, "Threading mode actual {} vs requested {}", s_threadingMode, required_mode);
      // @BUGBUG: for now I need to manually synchronize since my MPI version does not accept the threading mode parameter
      //ABORT_IF(
      //  threadSupportLevel_ < required_mode,
      //  "Your version of MPI does not support multi-threaded communication.");

      MPI_Comm_size(MPI_COMM_WORLD, &s_comm_world_size);
      MPI_Comm_rank(MPI_COMM_WORLD, &s_my_rank);
    }
    else { // already initialized; just make sure we got the right threading mode
      ABORT_IF(s_threadingMode != required_mode, "MPIWrapper constructed with inconsistent threading mode");
    }

    LOG(info, "MPI mode, node {} out of {}", s_my_rank, s_comm_world_size);
  }

  virtual size_t myRank()        const override { return (size_t)s_my_rank; };
  virtual size_t commWorldSize() const override { return (size_t)s_comm_world_size; };

  virtual void barrier(MPI_Comm comm) const override {
    std::lock_guard<std::mutex> guard(s_mutex); // @BUGBUG: for now I need to manually synchronize since my MPI version does not accept the threading mode parameter
    HANDLE_MPI_ERROR(MPI_Barrier(comm));
  }
  virtual void sSend(void* buf, size_t count, MPI_Datatype datatype, size_t destRank, int tag, MPI_Comm comm) const override {
    std::lock_guard<std::mutex> guard(s_mutex); // @BUGBUG: for now I need to manually synchronize since my MPI version does not accept the threading mode parameter
    HANDLE_MPI_ERROR(MPI_Ssend(buf, (int)count, datatype, (int)destRank, tag, comm));
  }
  virtual void recv(void* buf, size_t count, MPI_Datatype datatype, size_t sourceRank, int tag, MPI_Comm comm, MPI_Status* status) const override {
    std::lock_guard<std::mutex> guard(s_mutex); // @BUGBUG: for now I need to manually synchronize since my MPI version does not accept the threading mode parameter
    HANDLE_MPI_ERROR(MPI_Recv(buf, (int)count, datatype, (int)sourceRank, tag, comm, status));
  }
  virtual void allReduce(const void* sendbuf, void* recvbuf, size_t count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) const override {
    std::lock_guard<std::mutex> guard(s_mutex); // @BUGBUG: for now I need to manually synchronize since my MPI version does not accept the threading mode parameter
    HANDLE_MPI_ERROR(MPI_Allreduce(sendbuf, recvbuf, (int)count, datatype, op, comm));
  }

  virtual void finalize() override {
    std::lock_guard<std::mutex> guard(s_mutex);
    ABORT_IF(s_useCount == 0, "finalize called too many times");
    if (s_useCount == 1) // last call finalizes MPI, i.e. tells MPI that we sucessfully completed computation
      HANDLE_MPI_ERROR(MPI_Finalize());
    s_useCount--;
  }
};

/*static*/ size_t     MPIWrapper::s_useCount = 0;
/*static*/ int        MPIWrapper::s_my_rank = -1;
/*static*/ int        MPIWrapper::s_comm_world_size = -1;
/*static*/ std::mutex MPIWrapper::s_mutex;
/*static*/ int        MPIWrapper::s_threadingMode;
#endif

// dummy MPI wrapper that implements only one process without actual operations
class FakeMPIWrapper : public IMPIWrapper
{
public:
  FakeMPIWrapper(bool) {
    LOG(warn, "compiled without MPI support; using FakeMPIWrapper to allow debugging");
  }

  virtual size_t myRank() const override { return 0; };
  virtual size_t commWorldSize() const override { return 1; };

#pragma warning(push)
#pragma warning(disable: 4100) // unreferenced formal parameter
  virtual void barrier(MPI_Comm comm) const override { }
  virtual void sSend(void* buf, size_t count, MPI_Datatype datatype, size_t destRank, int tag, MPI_Comm comm) const override
  {
    ABORT("should not send data to ourselves in dummy mode");
  }
  virtual void recv(void* buf, size_t count, MPI_Datatype datatype, size_t sourceRank, int tag, MPI_Comm comm, MPI_Status* status) const override
  {
    ABORT("should not attempt to receive from ourselves in dummy mode");
  }
  virtual void allReduce(const void* sendbuf, void* recvbuf, size_t count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) const override
  {
    ABORT("should not attempt to all-reduce from ourselves in dummy mode"); // @TODO: yes, we should, for testing
  }
#pragma warning(push)

  virtual void finalize() override { }
};

// create instance of the MPI wrapper
Ptr<IMPIWrapper> createMPIWrapper(bool sync) {
  // @TODO: This will be extended in the future to create other types, e.g. NCCL and fake for debugging
#if MPI_FOUND
  return New<MPIWrapper>(sync);
#else
  return New<FakeMPIWrapper>(sync);
#endif
}

}  // namespace marian
