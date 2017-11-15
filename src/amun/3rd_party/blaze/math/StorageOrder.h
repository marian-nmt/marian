//=================================================================================================
/*!
//  \file blaze/math/StorageOrder.h
//  \brief Header file for the matrix storage order types
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

#ifndef _BLAZE_MATH_STORAGEORDER_H_
#define _BLAZE_MATH_STORAGEORDER_H_


namespace blaze {

//=================================================================================================
//
//  MATRIX STORAGE ORDER TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Storage order flag for row-major matrices.
//
// Via this flag it is possible to specify the storage order of matrices as row-major. For
// instance, given the following matrix

                          \f[\left(\begin{array}{*{3}{c}}
                          1 & 2 & 3 \\
                          4 & 5 & 6 \\
                          \end{array}\right)\f]\n

// in case of row-major order the elements are stored in the order

                          \f[\left(\begin{array}{*{6}{c}}
                          1 & 2 & 3 & 4 & 5 & 6. \\
                          \end{array}\right)\f]

// The following example demonstrates the setup of this \f$ 2 \times 3 \f$ matrix:

   \code
   using blaze::rowMajor;
   blaze::StaticMatrix<int,2UL,3UL,rowMajor> A( { { 1, 2, 3 }, { 4, 5, 6 } } );
   \endcode
*/
const bool rowMajor = false;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Storage order flag for column-major matrices.
//
// Via this flag it is possible to specify the storage order of matrices as column-major. For
// instance, given the following matrix

                          \f[\left(\begin{array}{*{3}{c}}
                          1 & 2 & 3 \\
                          4 & 5 & 6 \\
                          \end{array}\right)\f]\n

// in case of column-major order the elements are stored in the order

                          \f[\left(\begin{array}{*{6}{c}}
                          1 & 4 & 2 & 5 & 3 & 6. \\
                          \end{array}\right)\f]

// The following example demonstrates the setup of this \f$ 2 \times 3 \f$ matrix:

   \code
   using blaze::columnMajor;
   blaze::StaticMatrix<int,2UL,3UL,columnMajor> A( { { 1, 2, 3 }, { 4, 5, 6 } } );
   \endcode
*/
const bool columnMajor = true;
//*************************************************************************************************

} // namespace blaze

#endif
