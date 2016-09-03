//=================================================================================================
/*!
//  \file blaze/config/Optimizations.h
//  \brief Configuration of performance optimizations
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

//*************************************************************************************************
/*!\brief Configuration of the padding of dense vectors and matrices.
// \ingroup config
//
// This configuration switch enables/disables the padding of dense vectors and matrices. Padding
// is used by the Blaze library in order to achieve maximum performance for both dense vector
// and matrix operations. Due to padding, the proper alignment of data elements can be guaranteed
// and the need for remainder loops is minimized. In case the switch is set to \a true, padding
// is enabled for all native dense vectors and matrices. If the switch is set to \a false, padding
// is generally disabled.
//
// \warning Note that disabling padding can considerably reduce the performance of all dense
// vector and matrix operations!
*/
constexpr bool usePadding = true;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Configuration of the streaming behavior.
// \ingroup config
//
// For large vectors and matrices non-temporal stores can provide a significant performance
// advantage of about 20%. However, this advantage is only in effect in case the memory bandwidth
// of the target architecture is maxed out. If the target architecture's memory bandwidth cannot
// be exhausted the use of non-temporal stores can decrease performance instead of increasing it.
//
// Via this compilation switch streaming (i.e. non-temporal stores) can be (de-)activated. If
// set to \a true streaming is enabled, if set to \a false streaming is disabled.
*/
constexpr bool useStreaming = true;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Configuration switch for optimized kernels.
// \ingroup config
//
// This configuration switch enables/disables all optimized compute kernels of the Blaze library,
// including all vectorized and data type depending kernels. In case the switch is set to \a true
// the optimized kernels are used whenever possible. In case the switch is set to \a false all
// optimized kernels are not used, even if it would be possible.
//
// \warning Note that disabling the optimized kernels causes a severe performance limitiation
// to nearly all operations!
*/
constexpr bool useOptimizedKernels = true;
//*************************************************************************************************

} // namespace blaze
