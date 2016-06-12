# Copyright 2011-06-05 Susi Lehtola.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# 
# * Neither the names of Kitware, Inc., the Insight Software Consortium,
#   nor the names of their contributors may be used to endorse or promote
#   products derived from this software without specific prior written
#   permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Try to find libgsl.
# Once done, this will define
#
# GSL_FOUND - system has libgsl
# GSL_INCLUDE_DIRS - the libgsl include directories
# GSL_LIBRARIES - link these to use libgsl

include(LibFindMacros)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(GSL_PKGCONF gsl)

# Include dir
find_path(GSL_INCLUDE_DIR
 NAMES gsl/gsl_blas.h gsl/gsl_integration.h gsl/gsl_multimin.h gsl/gsl_rng.h gsl/gsl_sf_bessel.h gsl/gsl_sf_coupling.h gsl/gsl_sf_gamma.h gsl/gsl_sf_hyperg.h gsl/gsl_sf_legendre.h gsl/gsl_spline.h
 PATHS ${GSL_PKGCONF_INCLUDE_DIRS}
)

# Library
find_library(GSL_LIBRARY
 NAMES gsl
 PATHS ${GSL_PKGCONF_LIBRARY_DIRS}
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.

set(GSL_PROCESS_INCLUDES GSL_INCLUDE_DIR)
set(GSL_PROCESS_LIBS GSL_LIBRARY)
libfind_process(GSL)
