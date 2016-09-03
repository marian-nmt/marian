//=================================================================================================
/*!
//  \file blaze/config/Thresholds.h
//  \brief Configuration of the thresholds for matrix/vector and matrix/matrix multiplications
//
//  Copyright (C) 2013 Klaus Iglberger - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or modify it under
//  the terms of the New (Revised) BSD License. Redistribution and use in source and binary
//  forms, with or without modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice, this list
//     of conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//  3. Neither the names of the Blaze development group nor the names of its contributors
//     may be used to endorse or promote products derived from this software without specific
//     prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//  SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//  DAMAGE.
*/
//=================================================================================================


namespace blaze {

//=================================================================================================
//
//  BLAS THRESHOLDS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Row-major dense matrix/dense vector multiplication threshold.
// \ingroup config
//
// This setting specifies the threshold between the application of the custom Blaze kernels
// and the BLAS kernels for the row-major dense matrix/dense vector multiplication. In case
// the number of elements in the dense matrix is equal or higher than this value, the BLAS
// kernels are preferred over the custom Blaze kernels. In case the number of elements in the
// dense matrix is smaller, the Blaze kernels are used.
//
// The default setting for this threshold is 4000000 (which for instance corresponds to a matrix
// size of \f$ 2000 \times 2000 \f$). Note that in case the Blaze debug mode is active, this
// threshold will be replaced by the blaze::DMATDVECMULT_DEBUG_THRESHOLD value.
*/
constexpr size_t DMATDVECMULT_USER_THRESHOLD = 4000000UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Column-major dense matrix/dense vector multiplication threshold.
// \ingroup config
//
// This setting specifies the threshold between the application of the custom Blaze kernels
// and the BLAS kernels for the column-major dense matrix/dense vector multiplication. In case
// the number of elements in the dense matrix is equal or higher than this value, the BLAS
// kernels are preferred over the custom Blaze kernels. In case the number of elements in the
// dense matrix is smaller, the Blaze kernels are used.
//
// The default setting for this threshold is 62500 (which for instance corresponds to a matrix
// size of \f$ 250 \times 250 \f$). Note that in case the Blaze debug mode is active, this
// threshold will be replaced by the blaze::TDMATDVECMULT_DEBUG_THRESHOLD value.
*/
constexpr size_t TDMATDVECMULT_USER_THRESHOLD = 62500UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Dense Vector/row-major dense matrix multiplication threshold.
// \ingroup config
//
// This setting specifies the threshold between the application of the custom Blaze kernels
// and the BLAS kernels for the dense vector/row-major dense matrix multiplication. In case
// the number of elements in the dense matrix is equal or higher than this value, the BLAS
// kernels are preferred over the custom Blaze kernels. In case the number of elements in the
// dense matrix is smaller, the Blaze kernels are used.
//
// The default setting for this threshold is 62500 (which for instance corresponds to a matrix
// size of \f$ 250 \times 250 \f$). Note that in case the Blaze debug mode is active, this
// threshold will be replaced by the blaze::TDVECDMATMULT_DEBUG_THRESHOLD value.
*/
constexpr size_t TDVECDMATMULT_USER_THRESHOLD = 62500UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Dense Vector/column-major dense matrix multiplication threshold.
// \ingroup config
//
// This setting specifies the threshold between the application of the custom Blaze kernels
// and the BLAS kernels for the dense vector/column-major dense matrix multiplication. In case
// the number of elements in the dense matrix is equal or higher than this value, the BLAS
// kernels are preferred over the custom Blaze kernels. In case the number of elements in the
// dense matrix is smaller, the Blaze kernels are used.
//
// The default setting for this threshold is 4000000 (which for instance corresponds to a matrix
// size of \f$ 2000 \times 2000 \f$). Note that in case the Blaze debug mode is active, this
// threshold will be replaced by the blaze::TDVECTDMATMULT_DEBUG_THRESHOLD value.
*/
constexpr size_t TDVECTDMATMULT_USER_THRESHOLD = 4000000UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Row-major dense matrix/row-major dense matrix multiplication threshold.
// \ingroup config
//
// This setting specifies the threshold between the application of the custom Blaze kernels and
// the BLAS kernels for the row-major dense matrix/row-major dense matrix multiplication. In
// case the number of elements of the target matrix is equal or higher than this value, the
// BLAS kernels are preferred over the custom Blaze kernels. In case the number of elements in
// the target matrix is smaller, the Blaze kernels are used.
//
// The default setting for this threshold is 10000 (which for instance corresponds to a matrix
// size of \f$ 100 \times 100 \f$). Note that in case the Blaze debug mode is active, this
// threshold will be replaced by the blaze::DMATDMATMULT_DEBUG_THRESHOLD value.
*/
constexpr size_t DMATDMATMULT_USER_THRESHOLD = 10000UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Row-major dense matrix/column-major dense matrix multiplication threshold.
// \ingroup config
//
// This setting specifies the threshold between the application of the custom Blaze kernels and
// the BLAS kernels for the row-major dense matrix/column-major dense matrix multiplication. In
// case the number of elements of the target matrix is equal or higher than this value, the
// BLAS kernels are preferred over the custom Blaze kernels. In case the number of elements in
// the target matrix is smaller, the Blaze kernels are used.
//
// The default setting for this threshold is 10000 (which for instance corresponds to a matrix
// size of \f$ 100 \times 100 \f$). Note that in case the Blaze debug mode is active, this
// threshold will be replaced by the blaze::DMATTDMATMULT_DEBUG_THRESHOLD value.
*/
constexpr size_t DMATTDMATMULT_USER_THRESHOLD = 10000UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Column-major dense matrix/row-major dense matrix multiplication threshold.
// \ingroup config
//
// This setting specifies the threshold between the application of the custom Blaze kernels and
// the BLAS kernels for the column-major dense matrix/row-major dense matrix multiplication. In
// case the number of elements of the target matrix is equal or higher than this value, the
// BLAS kernels are preferred over the custom Blaze kernels. In case the number of elements in
// the target matrix is smaller, the Blaze kernels are used.
//
// The default setting for this threshold is 10000 (which for instance corresponds to a matrix
// size of \f$ 100 \times 100 \f$). Note that in case the Blaze debug mode is active, this
// threshold will be replaced by the blaze::TDMATDMATMULT_DEBUG_THRESHOLD value.
*/
constexpr size_t TDMATDMATMULT_USER_THRESHOLD = 10000UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Column-major dense matrix/column-major dense matrix multiplication threshold.
// \ingroup config
//
// This setting specifies the threshold between the application of the custom Blaze kernels and
// the BLAS kernels for the column-major dense matrix/column-major dense matrix multiplication.
// In case the number of elements of the target matrix is equal or higher than this value, the
// BLAS kernels are preferred over the custom Blaze kernels. In case the number of elements in
// the target matrix is smaller, the Blaze kernels are used.
//
// The default setting for this threshold is 10000 (which for instance corresponds to a matrix
// size of \f$ 100 \times 100 \f$). Note that in case the Blaze debug mode is active, this
// threshold will be replaced by the blaze::TDMATTDMATMULT_DEBUG_THRESHOLD value.
*/
constexpr size_t TDMATTDMATMULT_USER_THRESHOLD = 10000UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Row-major dense matrix/row-major sparse matrix multiplication threshold.
// \ingroup config
//
// This setting specifies the threshold between the application of the Blaze kernels for small
// and for large row-major dense matrix/row-major sparse matrix multiplications. In case the
// number of elements of the target matrix is equal or higher than this value, the kernel for
// large matrices is preferred over the kernel for small matrices. In case the number of elements
// in the target matrix is smaller, the kernel for small matrices is used.
//
// The default setting for this threshold is 2500 (which for instance corresponds to a matrix
// size of \f$ 50 \times 50 \f$). Note that in case the Blaze debug mode is active, this
// threshold will be replaced by the blaze::DMATSMATMULT_DEBUG_THRESHOLD value.
*/
constexpr size_t DMATSMATMULT_USER_THRESHOLD = 2500UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Column-major dense matrix/row-major sparse matrix multiplication threshold.
// \ingroup config
//
// This setting specifies the threshold between the application of the Blaze kernels for small
// and for large column-major dense matrix/row-major sparse matrix multiplications. In case the
// number of elements of the target matrix is equal or higher than this value, the kernel for
// large matrices is preferred over the kernel for small matrices. In case the number of elements
// in the target matrix is smaller, the kernel for small matrices is used.
//
// The default setting for this threshold is 2500 (which for instance corresponds to a matrix
// size of \f$ 50 \times 50 \f$). Note that in case the Blaze debug mode is active, this
// threshold will be replaced by the blaze::TDMATSMATMULT_DEBUG_THRESHOLD value.
*/
constexpr size_t TDMATSMATMULT_USER_THRESHOLD = 2500UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Column-major sparse matrix/row-major dense matrix multiplication threshold.
// \ingroup config
//
// This setting specifies the threshold between the application of the Blaze kernels for small
// and for large column-major sparse matrix/row-major dense matrix multiplications. In case the
// number of elements of the target matrix is equal or higher than this value, the kernel for
// large matrices is preferred over the kernel for small matrices. In case the number of elements
// in the target matrix is smaller, the kernel for small matrices is used.
//
// The default setting for this threshold is 10000 (which for instance corresponds to a matrix
// size of \f$ 100 \times 100 \f$). Note that in case the Blaze debug mode is active, this
// threshold will be replaced by the blaze::TSMATDMATMULT_DEBUG_THRESHOLD value.
*/
constexpr size_t TSMATDMATMULT_USER_THRESHOLD = 10000UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Column-major sparse matrix/column-major dense matrix multiplication threshold.
// \ingroup config
//
// This setting specifies the threshold between the application of the Blaze kernels for small
// and for large column-major sparse matrix/column-major dense matrix multiplications. In case
// the number of elements of the target matrix is equal or higher than this value, the kernel for
// large matrices is preferred over the kernel for small matrices. In case the number of elements
// in the target matrix is smaller, the kernel for small matrices is used.
//
// The default setting for this threshold is 22500 (which for instance corresponds to a matrix
// size of \f$ 150 \times 150 \f$). Note that in case the Blaze debug mode is active, this
// threshold will be replaced by the blaze::TSMATTDMATMULT_DEBUG_THRESHOLD value.
*/
constexpr size_t TSMATTDMATMULT_USER_THRESHOLD = 22500UL;
//*************************************************************************************************




//=================================================================================================
//
//  SMP THRESHOLDS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief SMP dense vector assignment threshold.
// \ingroup config
//
// This threshold specifies when an assignment of a simple dense vector can be executed in
// parallel. In case the number of elements of the target vector is larger or equal to this
// threshold, the operation is executed in parallel. If the number of elements is below this
// threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 38000. In case the threshold is set to 0, the
// operation is unconditionally executed in parallel.
*/
constexpr size_t SMP_DVECASSIGN_USER_THRESHOLD = 38000UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/dense vector addition threshold.
// \ingroup config
//
// This threshold specifies when a dense vector/dense vector addition can be executed in parallel.
// In case the number of elements of the target vector is larger or equal to this threshold, the
// operation is executed in parallel. If the number of elements is below this threshold the
// operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 38000. In case the threshold is set to 0, the
// operation is unconditionally executed in parallel.
*/
constexpr size_t SMP_DVECDVECADD_USER_THRESHOLD = 38000UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/dense vector subtraction threshold.
// \ingroup config
//
// This threshold specifies when a dense vector/dense vector subtraction can be executed in
// parallel. In case the number of elements of the target vector is larger or equal to this
// threshold, the operation is executed in parallel. If the number of elements is below this
// threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 38000. In case the threshold is set to 0, the
// operation is unconditionally executed in parallel.
*/
constexpr size_t SMP_DVECDVECSUB_USER_THRESHOLD = 38000UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/dense vector multiplication threshold.
// \ingroup config
//
// This threshold specifies when a dense vector/dense vector multiplication can be executed
// in parallel. In case the number of elements of the target vector is larger or equal to this
// threshold, the operation is executed in parallel. If the number of elements is below this
// threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 38000. In case the threshold is set to 0, the
// operation is unconditionally executed in parallel.
*/
constexpr size_t SMP_DVECDVECMULT_USER_THRESHOLD = 38000UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/dense vector division threshold.
// \ingroup config
//
// This threshold specifies when a dense vector/dense vector division can be executed in
// parallel. In case the number of elements of the target vector is larger or equal to this
// threshold, the operation is executed in parallel. If the number of elements is below this
// threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 38000. In case the threshold is set to 0, the
// operation is unconditionally executed in parallel.
*/
constexpr size_t SMP_DVECDVECDIV_USER_THRESHOLD = 38000UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/scalar multiplication/division threshold.
// \ingroup config
//
// This threshold specifies when a dense vector/scalar multiplication/division can be executed
// in parallel. In case the number of elements of the target vector is larger or equal to this
// threshold, the operation is executed in parallel. If the number of elements is below this
// threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 51000. In case the threshold is set to 0, the
// operation is unconditionally executed in parallel.
*/
constexpr size_t SMP_DVECSCALARMULT_USER_THRESHOLD = 51000UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/dense vector multiplication threshold.
// \ingroup config
//
// This threshold specifies when a row-major dense matrix/dense vector multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 330. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_DMATDVECMULT_USER_THRESHOLD = 330UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major dense matrix/dense vector multiplication threshold.
// \ingroup config
//
// This threshold specifies when a column-major dense matrix/dense vector multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 360. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TDMATDVECMULT_USER_THRESHOLD = 360UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/row-major dense matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a dense vector/row-major dense matrix multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 370. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TDVECDMATMULT_USER_THRESHOLD = 370UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/column-major dense matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a dense vector/column-major dense matrix multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 340. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TDVECTDMATMULT_USER_THRESHOLD = 340UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/sparse vector multiplication threshold.
// \ingroup config
//
// This threshold specifies when a row-major dense matrix/sparse vector multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 480. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_DMATSVECMULT_USER_THRESHOLD = 480UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major dense matrix/sparse vector multiplication threshold.
// \ingroup config
//
// This threshold specifies when a column-major dense matrix/sparse vector multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 910. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TDMATSVECMULT_USER_THRESHOLD = 910UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP sparse vector/row-major dense matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a sparse vector/row-major dense matrix multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 910. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TSVECDMATMULT_USER_THRESHOLD = 910UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP sparse vector/column-major dense matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a sparse vector/column-major dense matrix multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 480. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TSVECTDMATMULT_USER_THRESHOLD = 480UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major sparse matrix/dense vector multiplication threshold.
// \ingroup config
//
// This threshold specifies when a row-major sparse matrix/dense vector multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 600. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_SMATDVECMULT_USER_THRESHOLD = 600UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major sparse matrix/dense vector multiplication threshold.
// \ingroup config
//
// This threshold specifies when a column-major sparse matrix/dense vector multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 1250. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TSMATDVECMULT_USER_THRESHOLD = 1250UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/row-major sparse matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a dense vector/row-major sparse matrix multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 1190. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TDVECSMATMULT_USER_THRESHOLD = 1190UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/column-major sparse matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a dense vector/column-major sparse matrix multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 530. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TDVECTSMATMULT_USER_THRESHOLD = 530UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major sparse matrix/sparse vector multiplication threshold.
// \ingroup config
//
// This threshold specifies when a row-major sparse matrix/sparse vector multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 260. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_SMATSVECMULT_USER_THRESHOLD = 260UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major sparse matrix/sparse vector multiplication threshold.
// \ingroup config
//
// This threshold specifies when a column-major sparse matrix/sparse vector multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 2160. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TSMATSVECMULT_USER_THRESHOLD = 2160UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP sparse vector/row-major sparse matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a sparse vector/row-major sparse matrix multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 2160. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TSVECSMATMULT_USER_THRESHOLD = 2160UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP sparse vector/column-major sparse matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a sparse vector/column-major sparse matrix multiplication can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 260. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TSVECTSMATMULT_USER_THRESHOLD = 260UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense matrix assignment threshold.
// \ingroup config
//
// This threshold specifies when an assignment with a simple dense matrix can be executed in
// parallel. In case the number of rows/columns of the target matrix is larger or equal to this
// threshold, the operation is executed in parallel. If the number of rows/columns is below this
// threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 220. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_DMATASSIGN_USER_THRESHOLD = 220UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/row-major dense matrix addition threshold.
// \ingroup config
//
// This threshold specifies when a row-major dense matrix/row-major dense matrix addition can
// be executed in parallel. This threshold affects both additions between two row-major matrices
// or two column-major dense matrices. In case the number of rows/columns of the target matrix
// is larger or equal to this threshold, the operation is executed in parallel. If the number of
// rows/columns is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 190. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_DMATDMATADD_USER_THRESHOLD = 190UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/column-major dense matrix addition threshold.
// \ingroup config
//
// This threshold specifies when a row-major dense matrix/column-major dense matrix addition can
// be executed in parallel. This threshold affects both additions between a row-major matrix and
// a column-major matrix and a column-major matrix and a row-major matrix. In case the number of
// rows/columns of the target matrix is larger or equal to this threshold, the operation is
// executed in parallel. If the number of rows/columns is below this threshold the operation is
// executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 175. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_DMATTDMATADD_USER_THRESHOLD = 175UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/row-major dense matrix subtraction threshold.
// \ingroup config
//
// This threshold specifies when a row-major dense matrix/row-major dense matrix subtraction
// can be executed in parallel. This threshold affects both subtractions between two row-major
// matrices or two column-major dense matrices. In case the number of rows/columns of the target
// matrix is larger or equal to this threshold, the operation is executed in parallel. If the
// number of rows/columns is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 190. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_DMATDMATSUB_USER_THRESHOLD = 190UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/column-major dense matrix subtraction threshold.
// \ingroup config
//
// This threshold specifies when a row-major dense matrix/column-major dense matrix subtraction
// can be executed in parallel. This threshold affects both subtractions between a row-major
// matrix and a column-major matrix and a column-major matrix and a row-major matrix. In case
// the number of rows/columns of the target matrix is larger or equal to this threshold, the
// operation is executed in parallel. If the number of rows/columns is below this threshold
// the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 175. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_DMATTDMATSUB_USER_THRESHOLD = 175UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense matrix/scalar multiplication/division threshold.
// \ingroup config
//
// This threshold specifies when a dense matrix/scalar multiplication or division can be executed
// in parallel. In case the number of rows/columns of the target matrix is larger or equal to this
// threshold, the operation is executed in parallel. If the number of rows/columns is below this
// threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 220. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_DMATSCALARMULT_USER_THRESHOLD = 220UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/row-major dense matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a row-major dense matrix/row-major dense matrix multiplication
// can be executed in parallel. In case the number of rows/columns of the target matrix is larger
// or equal to this threshold, the operation is executed in parallel. If the number of rows/columns
// is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 55. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_DMATDMATMULT_USER_THRESHOLD = 55UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/column-major dense matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a row-major dense matrix/column-major dense matrix multiplication
// can be executed in parallel. In case the number of rows/columns of the target matrix is larger
// or equal to this threshold, the operation is executed in parallel. If the number of rows/columns
// is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 55. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_DMATTDMATMULT_USER_THRESHOLD = 55UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major dense matrix/row-major dense matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a column-major dense matrix/row-major dense matrix multiplication
// can be executed in parallel. In case the number of rows/columns of the target matrix is larger
// or equal to this threshold, the operation is executed in parallel. If the number of rows/columns
// is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 55. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TDMATDMATMULT_USER_THRESHOLD = 55UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major dense matrix/column-major dense matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a column-major dense matrix/column-major dense matrix multiplication
// can be executed in parallel. In case the number of rows/columns of the target matrix is larger
// or equal to this threshold, the operation is executed in parallel. If the number of rows/columns
// is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 55. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TDMATTDMATMULT_USER_THRESHOLD = 55UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/row-major sparse matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a row-major dense matrix/row-major sparse matrix multiplication
// can be executed in parallel. In case the number of rows/columns of the target matrix is larger
// or equal to this threshold, the operation is executed in parallel. If the number of rows/columns
// is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 64. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_DMATSMATMULT_USER_THRESHOLD = 64UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/column-major sparse matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a row-major dense matrix/column-major sparse matrix multiplication
// can be executed in parallel. In case the number of rows/columns of the target matrix is larger
// or equal to this threshold, the operation is executed in parallel. If the number of rows/columns
// is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 68. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_DMATTSMATMULT_USER_THRESHOLD = 68UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major dense matrix/row-major sparse matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a column-major dense matrix/row-major sparse matrix multiplication
// can be executed in parallel. In case the number of rows/columns of the target matrix is larger
// or equal to this threshold, the operation is executed in parallel. If the number of rows/columns
// is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 90. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TDMATSMATMULT_USER_THRESHOLD = 90UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major dense matrix/column-major sparse matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a column-major dense matrix/column-major sparse matrix multiplication
// can be executed in parallel. In case the number of rows/columns of the target matrix is larger
// or equal to this threshold, the operation is executed in parallel. If the number of rows/columns
// is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 90. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TDMATTSMATMULT_USER_THRESHOLD = 90UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major sparse matrix/row-major dense matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a row-major sparse matrix/row-major dense matrix multiplication
// can be executed in parallel. In case the number of rows/columns of the target matrix is larger
// or equal to this threshold, the operation is executed in parallel. If the number of rows/columns
// is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 88. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_SMATDMATMULT_USER_THRESHOLD = 88UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major sparse matrix/column-major dense matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a row-major sparse matrix/column-major dense matrix multiplication
// can be executed in parallel. In case the number of rows/columns of the target matrix is larger
// or equal to this threshold, the operation is executed in parallel. If the number of rows/columns
// is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 72. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_SMATTDMATMULT_USER_THRESHOLD = 72UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major sparse matrix/row-major dense matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a column-major sparse matrix/row-major dense matrix multiplication
// can be executed in parallel. In case the number of rows/columns of the target matrix is larger
// or equal to this threshold, the operation is executed in parallel. If the number of rows/columns
// is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 66. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TSMATDMATMULT_USER_THRESHOLD = 66UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major sparse matrix/column-major dense matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a column-major sparse matrix/column-major dense matrix multiplication
// can be executed in parallel. In case the number of rows/columns of the target matrix is larger
// or equal to this threshold, the operation is executed in parallel. If the number of rows/columns
// is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 66. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TSMATTDMATMULT_USER_THRESHOLD = 66UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major sparse matrix/row-major sparse matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a row-major sparse matrix/row-major sparse matrix multiplication
// can be executed in parallel. In case the number of rows/columns of the target matrix is larger
// or equal to this threshold, the operation is executed in parallel. If the number of rows/columns
// is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 150. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_SMATSMATMULT_USER_THRESHOLD = 150UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major sparse matrix/column-major sparse matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a row-major sparse matrix/column-major sparse matrix multiplication
// can be executed in parallel. In case the number of rows/columns of the target matrix is larger
// or equal to this threshold, the operation is executed in parallel. If the number of rows/columns
// is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 140. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_SMATTSMATMULT_USER_THRESHOLD = 140UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major sparse matrix/row-major sparse matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a column-major sparse matrix/row-major sparse matrix multiplication
// can be executed in parallel. In case the number of rows/columns of the target matrix is larger
// or equal to this threshold, the operation is executed in parallel. If the number of rows/columns
// is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 140. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TSMATSMATMULT_USER_THRESHOLD = 140UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major sparse matrix/column-major sparse matrix multiplication threshold.
// \ingroup config
//
// This threshold specifies when a column-major sparse matrix/column-major sparse matrix
// multiplication can be executed in parallel. In case the number of rows/columns of the target
// matrix is larger or equal to this threshold, the operation is executed in parallel. If the
// number of rows/columns is below this threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 150. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_TSMATTSMATMULT_USER_THRESHOLD = 150UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/dense vector outer product threshold.
// \ingroup config
//
// This threshold specifies when a dense vector/dense vector outer product can be executed in
// parallel. In case the number of rows/columns of the target matrix is larger or equal to this
// threshold, the operation is executed in parallel. If the number of rows/columns is below this
// threshold the operation is executed single-threaded.
//
// Please note that this threshold is highly sensitiv to the used system architecture and the
// shared memory parallelization technique. Therefore the default value cannot guarantee maximum
// performance for all possible situations and configurations. It merely provides a reasonable
// standard for the current generation of CPUs. Also note that the provided default has been
// determined using the OpenMP parallelization and requires individual adaption for the C++11
// and Boost thread parallelization.
//
// The default setting for this threshold is 290. In case the threshold is set to 0, the operation
// is unconditionally executed in parallel.
*/
constexpr size_t SMP_DVECTDVECMULT_USER_THRESHOLD = 290UL;
//*************************************************************************************************

} // namespace blaze
