//=================================================================================================
/*!
//  \file blaze/math/Traits.h
//  \brief Header file for all expression traits
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

#ifndef _BLAZE_MATH_TRAITS_H_
#define _BLAZE_MATH_TRAITS_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/traits/AddExprTrait.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/ColumnTrait.h>
#include <blaze/math/traits/CrossExprTrait.h>
#include <blaze/math/traits/CrossTrait.h>
#include <blaze/math/traits/CTransExprTrait.h>
#include <blaze/math/traits/DerestrictTrait.h>
#include <blaze/math/traits/DivExprTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/DMatCTransExprTrait.h>
#include <blaze/math/traits/DMatDMatAddExprTrait.h>
#include <blaze/math/traits/DMatDMatMultExprTrait.h>
#include <blaze/math/traits/DMatDMatSubExprTrait.h>
#include <blaze/math/traits/DMatDVecMultExprTrait.h>
#include <blaze/math/traits/DMatEvalExprTrait.h>
#include <blaze/math/traits/DMatForEachExprTrait.h>
#include <blaze/math/traits/DMatInvExprTrait.h>
#include <blaze/math/traits/DMatScalarDivExprTrait.h>
#include <blaze/math/traits/DMatScalarMultExprTrait.h>
#include <blaze/math/traits/DMatSerialExprTrait.h>
#include <blaze/math/traits/DMatSMatAddExprTrait.h>
#include <blaze/math/traits/DMatSMatMultExprTrait.h>
#include <blaze/math/traits/DMatSMatSubExprTrait.h>
#include <blaze/math/traits/DMatSVecMultExprTrait.h>
#include <blaze/math/traits/DMatTDMatAddExprTrait.h>
#include <blaze/math/traits/DMatTDMatMultExprTrait.h>
#include <blaze/math/traits/DMatTDMatSubExprTrait.h>
#include <blaze/math/traits/DMatTransExprTrait.h>
#include <blaze/math/traits/DMatTSMatAddExprTrait.h>
#include <blaze/math/traits/DMatTSMatMultExprTrait.h>
#include <blaze/math/traits/DMatTSMatSubExprTrait.h>
#include <blaze/math/traits/DVecCTransExprTrait.h>
#include <blaze/math/traits/DVecDVecAddExprTrait.h>
#include <blaze/math/traits/DVecDVecCrossExprTrait.h>
#include <blaze/math/traits/DVecDVecDivExprTrait.h>
#include <blaze/math/traits/DVecDVecMultExprTrait.h>
#include <blaze/math/traits/DVecDVecSubExprTrait.h>
#include <blaze/math/traits/DVecEvalExprTrait.h>
#include <blaze/math/traits/DVecForEachExprTrait.h>
#include <blaze/math/traits/DVecScalarDivExprTrait.h>
#include <blaze/math/traits/DVecScalarMultExprTrait.h>
#include <blaze/math/traits/DVecSerialExprTrait.h>
#include <blaze/math/traits/DVecSVecAddExprTrait.h>
#include <blaze/math/traits/DVecSVecCrossExprTrait.h>
#include <blaze/math/traits/DVecSVecMultExprTrait.h>
#include <blaze/math/traits/DVecSVecSubExprTrait.h>
#include <blaze/math/traits/DVecTDVecMultExprTrait.h>
#include <blaze/math/traits/DVecTransExprTrait.h>
#include <blaze/math/traits/DVecTSVecMultExprTrait.h>
#include <blaze/math/traits/EvalExprTrait.h>
#include <blaze/math/traits/ForEachExprTrait.h>
#include <blaze/math/traits/ForEachTrait.h>
#include <blaze/math/traits/ImagTrait.h>
#include <blaze/math/traits/InvExprTrait.h>
#include <blaze/math/traits/MathTrait.h>
#include <blaze/math/traits/MultExprTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/RealTrait.h>
#include <blaze/math/traits/RowTrait.h>
#include <blaze/math/traits/SerialExprTrait.h>
#include <blaze/math/traits/SMatCTransExprTrait.h>
#include <blaze/math/traits/SMatDMatAddExprTrait.h>
#include <blaze/math/traits/SMatDMatMultExprTrait.h>
#include <blaze/math/traits/SMatDMatSubExprTrait.h>
#include <blaze/math/traits/SMatDVecMultExprTrait.h>
#include <blaze/math/traits/SMatEvalExprTrait.h>
#include <blaze/math/traits/SMatForEachExprTrait.h>
#include <blaze/math/traits/SMatScalarDivExprTrait.h>
#include <blaze/math/traits/SMatScalarMultExprTrait.h>
#include <blaze/math/traits/SMatSerialExprTrait.h>
#include <blaze/math/traits/SMatSMatAddExprTrait.h>
#include <blaze/math/traits/SMatSMatMultExprTrait.h>
#include <blaze/math/traits/SMatSMatSubExprTrait.h>
#include <blaze/math/traits/SMatSVecMultExprTrait.h>
#include <blaze/math/traits/SMatTDMatAddExprTrait.h>
#include <blaze/math/traits/SMatTDMatMultExprTrait.h>
#include <blaze/math/traits/SMatTDMatSubExprTrait.h>
#include <blaze/math/traits/SMatTransExprTrait.h>
#include <blaze/math/traits/SMatTSMatAddExprTrait.h>
#include <blaze/math/traits/SMatTSMatMultExprTrait.h>
#include <blaze/math/traits/SMatTSMatSubExprTrait.h>
#include <blaze/math/traits/SubExprTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/traits/SubvectorExprTrait.h>
#include <blaze/math/traits/SubvectorTrait.h>
#include <blaze/math/traits/SVecCTransExprTrait.h>
#include <blaze/math/traits/SVecDVecAddExprTrait.h>
#include <blaze/math/traits/SVecDVecCrossExprTrait.h>
#include <blaze/math/traits/SVecDVecDivExprTrait.h>
#include <blaze/math/traits/SVecDVecMultExprTrait.h>
#include <blaze/math/traits/SVecDVecSubExprTrait.h>
#include <blaze/math/traits/SVecEvalExprTrait.h>
#include <blaze/math/traits/SVecForEachExprTrait.h>
#include <blaze/math/traits/SVecScalarDivExprTrait.h>
#include <blaze/math/traits/SVecScalarMultExprTrait.h>
#include <blaze/math/traits/SVecSerialExprTrait.h>
#include <blaze/math/traits/SVecSVecAddExprTrait.h>
#include <blaze/math/traits/SVecSVecCrossExprTrait.h>
#include <blaze/math/traits/SVecSVecMultExprTrait.h>
#include <blaze/math/traits/SVecSVecSubExprTrait.h>
#include <blaze/math/traits/SVecTDVecMultExprTrait.h>
#include <blaze/math/traits/SVecTransExprTrait.h>
#include <blaze/math/traits/SVecTSVecMultExprTrait.h>
#include <blaze/math/traits/TDMatCTransExprTrait.h>
#include <blaze/math/traits/TDMatDMatAddExprTrait.h>
#include <blaze/math/traits/TDMatDMatMultExprTrait.h>
#include <blaze/math/traits/TDMatDMatSubExprTrait.h>
#include <blaze/math/traits/TDMatEvalExprTrait.h>
#include <blaze/math/traits/TDMatForEachExprTrait.h>
#include <blaze/math/traits/TDMatInvExprTrait.h>
#include <blaze/math/traits/TDMatScalarDivExprTrait.h>
#include <blaze/math/traits/TDMatScalarMultExprTrait.h>
#include <blaze/math/traits/TDMatSerialExprTrait.h>
#include <blaze/math/traits/TDMatSMatAddExprTrait.h>
#include <blaze/math/traits/TDMatSMatMultExprTrait.h>
#include <blaze/math/traits/TDMatSMatSubExprTrait.h>
#include <blaze/math/traits/TDMatDVecMultExprTrait.h>
#include <blaze/math/traits/TDMatSVecMultExprTrait.h>
#include <blaze/math/traits/TDMatTDMatAddExprTrait.h>
#include <blaze/math/traits/TDMatTDMatMultExprTrait.h>
#include <blaze/math/traits/TDMatTDMatSubExprTrait.h>
#include <blaze/math/traits/TDMatTransExprTrait.h>
#include <blaze/math/traits/TDMatTSMatAddExprTrait.h>
#include <blaze/math/traits/TDMatTSMatMultExprTrait.h>
#include <blaze/math/traits/TDMatTSMatSubExprTrait.h>
#include <blaze/math/traits/TDVecCTransExprTrait.h>
#include <blaze/math/traits/TDVecDMatMultExprTrait.h>
#include <blaze/math/traits/TDVecDVecMultExprTrait.h>
#include <blaze/math/traits/TDVecEvalExprTrait.h>
#include <blaze/math/traits/TDVecForEachExprTrait.h>
#include <blaze/math/traits/TDVecScalarDivExprTrait.h>
#include <blaze/math/traits/TDVecScalarMultExprTrait.h>
#include <blaze/math/traits/TDVecSerialExprTrait.h>
#include <blaze/math/traits/TDVecSMatMultExprTrait.h>
#include <blaze/math/traits/TDVecSVecMultExprTrait.h>
#include <blaze/math/traits/TDVecTDMatMultExprTrait.h>
#include <blaze/math/traits/TDVecTDVecAddExprTrait.h>
#include <blaze/math/traits/TDVecTDVecCrossExprTrait.h>
#include <blaze/math/traits/TDVecTDVecDivExprTrait.h>
#include <blaze/math/traits/TDVecTDVecMultExprTrait.h>
#include <blaze/math/traits/TDVecTDVecSubExprTrait.h>
#include <blaze/math/traits/TDVecTransExprTrait.h>
#include <blaze/math/traits/TDVecTSMatMultExprTrait.h>
#include <blaze/math/traits/TDVecTSVecAddExprTrait.h>
#include <blaze/math/traits/TDVecTSVecCrossExprTrait.h>
#include <blaze/math/traits/TDVecTSVecMultExprTrait.h>
#include <blaze/math/traits/TDVecTSVecSubExprTrait.h>
#include <blaze/math/traits/TransExprTrait.h>
#include <blaze/math/traits/TSMatCTransExprTrait.h>
#include <blaze/math/traits/TSMatDMatAddExprTrait.h>
#include <blaze/math/traits/TSMatDMatMultExprTrait.h>
#include <blaze/math/traits/TSMatDMatSubExprTrait.h>
#include <blaze/math/traits/TSMatEvalExprTrait.h>
#include <blaze/math/traits/TSMatForEachExprTrait.h>
#include <blaze/math/traits/TSMatScalarDivExprTrait.h>
#include <blaze/math/traits/TSMatScalarMultExprTrait.h>
#include <blaze/math/traits/TSMatSerialExprTrait.h>
#include <blaze/math/traits/TSMatSMatAddExprTrait.h>
#include <blaze/math/traits/TSMatSMatMultExprTrait.h>
#include <blaze/math/traits/TSMatSMatSubExprTrait.h>
#include <blaze/math/traits/TSMatDVecMultExprTrait.h>
#include <blaze/math/traits/TSMatSVecMultExprTrait.h>
#include <blaze/math/traits/TSMatTDMatAddExprTrait.h>
#include <blaze/math/traits/TSMatTDMatMultExprTrait.h>
#include <blaze/math/traits/TSMatTDMatSubExprTrait.h>
#include <blaze/math/traits/TSMatTransExprTrait.h>
#include <blaze/math/traits/TSMatTSMatAddExprTrait.h>
#include <blaze/math/traits/TSMatTSMatMultExprTrait.h>
#include <blaze/math/traits/TSMatTSMatSubExprTrait.h>
#include <blaze/math/traits/TSVecCTransExprTrait.h>
#include <blaze/math/traits/TSVecDMatMultExprTrait.h>
#include <blaze/math/traits/TSVecDVecMultExprTrait.h>
#include <blaze/math/traits/TSVecEvalExprTrait.h>
#include <blaze/math/traits/TSVecForEachExprTrait.h>
#include <blaze/math/traits/TSVecScalarDivExprTrait.h>
#include <blaze/math/traits/TSVecScalarMultExprTrait.h>
#include <blaze/math/traits/TSVecSerialExprTrait.h>
#include <blaze/math/traits/TSVecSMatMultExprTrait.h>
#include <blaze/math/traits/TSVecSVecMultExprTrait.h>
#include <blaze/math/traits/TSVecTDMatMultExprTrait.h>
#include <blaze/math/traits/TSVecTDVecAddExprTrait.h>
#include <blaze/math/traits/TSVecTDVecCrossExprTrait.h>
#include <blaze/math/traits/TSVecTDVecDivExprTrait.h>
#include <blaze/math/traits/TSVecTDVecMultExprTrait.h>
#include <blaze/math/traits/TSVecTDVecSubExprTrait.h>
#include <blaze/math/traits/TSVecTransExprTrait.h>
#include <blaze/math/traits/TSVecTSMatMultExprTrait.h>
#include <blaze/math/traits/TSVecTSVecAddExprTrait.h>
#include <blaze/math/traits/TSVecTSVecCrossExprTrait.h>
#include <blaze/math/traits/TSVecTSVecMultExprTrait.h>
#include <blaze/math/traits/TSVecTSVecSubExprTrait.h>

#endif
