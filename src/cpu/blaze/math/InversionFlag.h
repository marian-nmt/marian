//=================================================================================================
/*!
//  \file blaze/math/InversionFlag.h
//  \brief Header file for the dense matrix inversion flags
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

#ifndef _BLAZE_MATH_INVERSIONFLAG_H_
#define _BLAZE_MATH_INVERSIONFLAG_H_


namespace blaze {

//=================================================================================================
//
//  DECOMPOSITION FLAG VALUES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Inversion flag.
// \ingroup dense_matrix
//
// The InversionFlag type enumeration represents the different types of matrix inversion algorithms
// that are available within the Blaze library. The following flags are available:
//
//  - \a byDefault: The default algorithm for each type of matrix. In case of general square
//          matrices an LU decomposition is used, in case of symmetric and Hermitian matrices
//          the Bunch-Kaufman diagonal pivoting method is applied, and in case of triangular
//          matrices a direct inversion via backward substitution is performed.
//  - \a byLU: The default inversion algorithm for general square matrices. It uses the LU
//          algorithm to decompose a matrix into a lower unitriangular matrix \c L, an upper
//          triangular matrix \c U, and a permutation matrix \c P (\f$ A = P L U \f$). If no
//          permutations are required, \c P is the identity matrix.
//  - \a byLDLT: The Bunch-Kaufman inversion algorithm for symmetric indefinite matrices. It
//          decomposes the given matrix into either \f$ A = U D U^{T} \f$ or \f$ A = L D L^{T} \f$,
//          where \c U (or \c L) is a product of permutation and unit upper (lower) triangular
//          matrices, and \c D is symmetric and block diagonal with 1-by-1 and 2-by-2 diagonal
//          blocks.
//  - \a byLDLH: The Bunch-Kaufman inversion algorithm for Hermitian indefinite matrices. It
//          decomposes the given matrix into either \f$ A = U D U^{H} \f$ or \f$ A = L D L^{H} \f$,
//          where \c U (or \c L) is a product of permutation and unit upper (lower) triangular
//          matrices, and \c D is Hermitian and block diagonal with 1-by-1 and 2-by-2 diagonal
//          blocks.
//  - \a byLLH: The Cholesky inversion algorithm for Hermitian positive definite matrices. It
//          decomposes a given matrix into either \f$ A = L L^H \f$, where \c L is a lower
//          triangular matrix, or \f$ A = U^H U \f$, where \c U is an upper triangular matrix.
*/
enum InversionFlag
{
   byDefault = 0,  //!< Flag for the default, optimal inversion algorithm.
   byLU      = 1,  //!< Flag for the LU-based matrix inversion.
   byLDLT    = 2,  //!< Flag for the Bunch-Kaufman-based inversion for symmetric matrices.
   byLDLH    = 3,  //!< Flag for the Bunch-Kaufman-based inversion for Hermitian matrices.
   byLLH     = 4   //!< Flag for the Cholesky-based inversion for positive-definite matrices.
};
//*************************************************************************************************

} // namespace blaze

#endif
