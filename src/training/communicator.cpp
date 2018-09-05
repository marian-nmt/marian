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
  int my_rank_;         // MPI rank of this node
  int comm_world_size_; // Number of nodes in MPI world (cluster)

public:
  MPIWrapper(bool sync) {
    int required_mode = sync ? MPI_THREAD_SERIALIZED : MPI_THREAD_MULTIPLE;
    int provided_thread_mode = 0;
    int argc = 1;
    char* argv[] = { "this.exe" };
    char** argvp = argv;
    MPI_Init_thread(&argc, &argvp, MPI_THREAD_MULTIPLE, &provided_thread_mode);
    // Enable if occasional truncation errors
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    ABORT_IF(
      provided_thread_mode < required_mode,
      "Your version of MPI does not support multi-threaded communication.");

    MPI_Comm_size(MPI_COMM_WORLD, &comm_world_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);
  }

  virtual size_t myRank() const override { return (size_t)my_rank_; };
  virtual size_t commWorldSize() const override { return (size_t)comm_world_size_; };

  // @TODO: error handling??
  virtual void barrier(MPI_Comm comm) const override
  {
    MPI_Barrier(comm);
  }
  virtual void sSend(void* buf, size_t count, MPI_Datatype datatype, size_t destRank, int tag, MPI_Comm comm) const override
  {
    MPI_Ssend(buf, (int)count, datatype, (int)destRank, tag, comm);
  }
  virtual void recv(void* buf, size_t count, MPI_Datatype datatype, size_t sourceRank, int tag, MPI_Comm comm, MPI_Status* status) const override
  {
    MPI_Recv(buf, (int)count, datatype, (int)sourceRank, tag, comm, status);
  }
  virtual void allReduce(const void* sendbuf, void* recvbuf, size_t count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) const override
  {
    MPI_Allreduce(sendbuf, recvbuf, (int)count, datatype, op, comm);
  }

  virtual void finalize() override { MPI_Finalize(); }
};
#endif

// create instance of the MPI wrapper
Ptr<IMPIWrapper> createMPIWrapper(bool sync) {
  // @TODO: This will be extended in the future to create other types, e.g. NCCL and fake for debugging
#if MPI_FOUND
  return New<MPIWrapper>(sync);
#else
  ABORT("MPI not found.");
#endif
}

}  // namespace marian
