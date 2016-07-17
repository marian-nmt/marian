# Copyright 2006-2008 The FLWOR Foundation.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# - Find Google Perf Tools
#
# GooglePerfTools_LIBRARIES - List of libraries when using Google Perf tools.
# GooglePerfTools_FOUND - True if Google Perf tools found.
 
 
SET(GooglePerfTools_NAMES profiler)
FIND_LIBRARY(GooglePerfTools_LIBRARY
  NAMES ${GooglePerfTools_NAMES}
  PATHS /usr/lib /usr/local/lib /opt/local/lib
)
 
IF (GooglePerfTools_LIBRARY)
   SET(GooglePerfTools_FOUND TRUE)
   SET( GooglePerfTools_LIBRARIES ${GooglePerfTools_LIBRARY} )
ELSE (GooglePerfTools_LIBRARY)
   SET(GooglePerfTools_FOUND FALSE)
   SET( GooglePerfTools_LIBRARIES )
ENDIF (GooglePerfTools_LIBRARY)
 
IF (GooglePerfTools_FOUND)
      MESSAGE(STATUS "Found GooglePerfTools: ${GooglePerfTools_LIBRARY}")
ELSE (GooglePerfTools_FOUND)
      MESSAGE(STATUS "Not Found GooglePerfTools: ${GooglePerfTools_LIBRARY}")
ENDIF (GooglePerfTools_FOUND)
 
MARK_AS_ADVANCED(
  GooglePerfTools_LIBRARY
)
 
