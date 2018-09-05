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

    //MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_world_size_);
    //MPI_Comm_rank(MPI_COMM_WORLD, &mpi_my_rank_);
  }

  virtual void finalize() override {
    MPI_Finalize();
  }
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
