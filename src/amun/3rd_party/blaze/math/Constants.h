//=================================================================================================
/*!
//  \file blaze/math/Constants.h
//  \brief Header file for mathematical constants
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

#ifndef _BLAZE_MATH_CONSTANTS_H_
#define _BLAZE_MATH_CONSTANTS_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cmath>
#include <blaze/system/Precision.h>


namespace blaze {

//=================================================================================================
//
//  MATHEMATICAL CONSTANT E
//
//=================================================================================================

#ifdef M_E
#  undef M_E
#endif

//*************************************************************************************************
/*!\brief Definition of the mathematical constant \f$ e \f$.
// \ingroup math
*/
const real_t M_E = 2.7182818284590452353602874713526625;
//*************************************************************************************************




//=================================================================================================
//
//  MATHEMATICAL CONSTANT LOG2E
//
//=================================================================================================

#ifdef M_LOG2E
#  undef M_LOG2E
#endif

//*************************************************************************************************
/*!\brief Definition of the mathematical constant \f$ \log_2 e \f$.
// \ingroup math
*/
const real_t M_LOG2E = 1.4426950408889634073599246810018921;
//*************************************************************************************************




//=================================================================================================
//
//  MATHEMATICAL CONSTANT LOG10E
//
//=================================================================================================

#ifdef M_LOG10E
#  undef M_LOG10E
#endif

//*************************************************************************************************
/*!\brief Definition of the mathematical constant \f$ \log_{10} e \f$.
// \ingroup math
*/
const real_t M_LOG10E = 0.4342944819032518276511289189166051;
//*************************************************************************************************




//=================================================================================================
//
//  MATHEMATICAL CONSTANT LN2
//
//=================================================================================================

#ifdef M_LN2
#  undef M_LN2
#endif

//*************************************************************************************************
/*!\brief Definition of the mathematical constant \f$ \ln 2 \f$.
// \ingroup math
*/
const real_t M_LN2 = 0.6931471805599453094172321214581766;
//*************************************************************************************************




//=================================================================================================
//
//  MATHEMATICAL CONSTANT LN10
//
//=================================================================================================

#ifdef M_LN10
#  undef M_LN10
#endif

//*************************************************************************************************
/*!\brief Definition of the mathematical constant \f$ \ln 10 \f$.
// \ingroup math
*/
const real_t M_LN10 = 2.3025850929940456840179914546843642;
//*************************************************************************************************




//=================================================================================================
//
//  MATHEMATICAL CONSTANT PI
//
//=================================================================================================

#ifdef M_PI
#  undef M_PI
#endif

//*************************************************************************************************
/*!\brief Definition of the mathematical constant \f$ \pi \f$.
// \ingroup math
*/
const real_t M_PI = 3.1415926535897932384626433832795029;
//*************************************************************************************************




//=================================================================================================
//
//  MATHEMATICAL CONSTANT SQRT2
//
//=================================================================================================

#ifdef M_SQRT2
#  undef M_SQRT2
#endif

//*************************************************************************************************
/*!\brief Definition of the mathematical constant \f$ \sqrt{2} \f$.
// \ingroup math
*/
const real_t M_SQRT2 = 1.4142135623730950488016887242096981;
//*************************************************************************************************




//=================================================================================================
//
//  MATHEMATICAL CONSTANT SQRT3
//
//=================================================================================================

#ifdef M_SQRT3
#  undef M_SQRT3
#endif

//*************************************************************************************************
/*!\brief Definition of the mathematical constant \f$ \sqrt{3} \f$.
// \ingroup math
*/
const real_t M_SQRT3 = 1.7320508075688772935274463415058724;
//*************************************************************************************************

} // namespace blaze

#endif
