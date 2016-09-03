//=================================================================================================
/*!
//  \file blaze/math/smp/default/SparseMatrix.h
//  \brief Header file for the default sparse matrix SMP implementation
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

#ifndef _BLAZE_MATH_SMP_DEFAULT_SPARSEMATRIX_H_
#define _BLAZE_MATH_SMP_DEFAULT_SPARSEMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/Matrix.h>
#include <blaze/math/expressions/SparseMatrix.h>
#include <blaze/math/typetraits/IsSparseMatrix.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/logging/FunctionTrace.h>


namespace blaze {

//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Sparse matrix SMP functions */
//@{
template< typename MT1, bool SO1, typename MT2, bool SO2 >
inline EnableIf_< IsSparseMatrix<MT1> >
   smpAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs );

template< typename MT1, bool SO1, typename MT2, bool SO2 >
inline EnableIf_< IsSparseMatrix<MT1> >
   smpAddAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs );

template< typename MT1, bool SO1, typename MT2, bool SO2 >
inline EnableIf_< IsSparseMatrix<MT1> >
   smpSubAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the SMP assignment of a matrix to a sparse matrix.
// \ingroup smp
//
// \param lhs The target left-hand side sparse matrix.
// \param rhs The right-hand side matrix to be assigned.
// \return void
//
// This function implements the default SMP assignment of a matrix to a sparse matrix.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side sparse matrix
        , bool SO1      // Storage order of the left-hand side sparse matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline EnableIf_< IsSparseMatrix<MT1> >
   smpAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );

   assign( ~lhs, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the SMP addition assignment of a matrix to a sparse matrix.
// \ingroup smp
//
// \param lhs The target left-hand side sparse matrix.
// \param rhs The right-hand side matrix to be added.
// \return void
//
// This function implements the default SMP addition assignment of a matrix to a sparse matrix.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side sparse matrix
        , bool SO1      // Storage order of the left-hand side sparse matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline EnableIf_< IsSparseMatrix<MT1> >
   smpAddAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );

   addAssign( ~lhs, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the SMP subtraction assignment of a matrix to sparse matrix.
// \ingroup smp
//
// \param lhs The target left-hand side sparse matrix.
// \param rhs The right-hand side matrix to be subtracted.
// \return void
//
// This function implements the default SMP subtraction assignment of a matrix to a sparse matrix.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side sparse matrix
        , bool SO1      // Storage order of the left-hand side sparse matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline EnableIf_< IsSparseMatrix<MT1> >
   smpSubAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );

   subAssign( ~lhs, ~rhs );
}
//*************************************************************************************************

} // namespace blaze

#endif
