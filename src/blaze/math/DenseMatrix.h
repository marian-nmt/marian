//=================================================================================================
/*!
//  \file blaze/math/DenseMatrix.h
//  \brief Header file for all basic DenseMatrix functionality
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

#ifndef _BLAZE_MATH_DENSEMATRIX_H_
#define _BLAZE_MATH_DENSEMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/dense/DenseMatrix.h>
#include <blaze/math/dense/Inversion.h>
#include <blaze/math/dense/LLH.h>
#include <blaze/math/dense/LQ.h>
#include <blaze/math/dense/LU.h>
#include <blaze/math/dense/QL.h>
#include <blaze/math/dense/QR.h>
#include <blaze/math/dense/RQ.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/DMatDetExpr.h>
#include <blaze/math/expressions/DMatDMatAddExpr.h>
#include <blaze/math/expressions/DMatDMatMultExpr.h>
#include <blaze/math/expressions/DMatDMatSubExpr.h>
#include <blaze/math/expressions/DMatDVecMultExpr.h>
#include <blaze/math/expressions/DMatEvalExpr.h>
#include <blaze/math/expressions/DMatForEachExpr.h>
#include <blaze/math/expressions/DMatInvExpr.h>
#include <blaze/math/expressions/DMatScalarDivExpr.h>
#include <blaze/math/expressions/DMatScalarMultExpr.h>
#include <blaze/math/expressions/DMatSerialExpr.h>
#include <blaze/math/expressions/DMatSMatAddExpr.h>
#include <blaze/math/expressions/DMatSMatMultExpr.h>
#include <blaze/math/expressions/DMatSMatSubExpr.h>
#include <blaze/math/expressions/DMatSVecMultExpr.h>
#include <blaze/math/expressions/DMatTDMatAddExpr.h>
#include <blaze/math/expressions/DMatTDMatMultExpr.h>
#include <blaze/math/expressions/DMatTDMatSubExpr.h>
#include <blaze/math/expressions/DMatTransExpr.h>
#include <blaze/math/expressions/DMatTransposer.h>
#include <blaze/math/expressions/DMatTSMatAddExpr.h>
#include <blaze/math/expressions/DMatTSMatMultExpr.h>
#include <blaze/math/expressions/DMatTSMatSubExpr.h>
#include <blaze/math/expressions/DVecTDVecMultExpr.h>
#include <blaze/math/expressions/SMatDMatMultExpr.h>
#include <blaze/math/expressions/SMatDMatSubExpr.h>
#include <blaze/math/expressions/SMatTDMatMultExpr.h>
#include <blaze/math/expressions/SMatTDMatSubExpr.h>
#include <blaze/math/expressions/SparseMatrix.h>
#include <blaze/math/expressions/TDMatDMatMultExpr.h>
#include <blaze/math/expressions/TDMatDVecMultExpr.h>
#include <blaze/math/expressions/TDMatSMatAddExpr.h>
#include <blaze/math/expressions/TDMatSMatMultExpr.h>
#include <blaze/math/expressions/TDMatSMatSubExpr.h>
#include <blaze/math/expressions/TDMatSVecMultExpr.h>
#include <blaze/math/expressions/TDMatTDMatMultExpr.h>
#include <blaze/math/expressions/TDMatTSMatMultExpr.h>
#include <blaze/math/expressions/TDVecDMatMultExpr.h>
#include <blaze/math/expressions/TDVecTDMatMultExpr.h>
#include <blaze/math/expressions/TSMatDMatMultExpr.h>
#include <blaze/math/expressions/TSMatDMatSubExpr.h>
#include <blaze/math/expressions/TSMatTDMatMultExpr.h>
#include <blaze/math/expressions/TSVecDMatMultExpr.h>
#include <blaze/math/expressions/TSVecTDMatMultExpr.h>
#include <blaze/math/Matrix.h>
#include <blaze/math/smp/DenseMatrix.h>
#include <blaze/math/smp/SparseMatrix.h>
#include <blaze/math/views/Column.h>
#include <blaze/math/views/Row.h>
#include <blaze/math/views/Submatrix.h>

#endif
