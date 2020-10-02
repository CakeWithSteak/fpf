#  Copyright (c) 2015 Light Transport Entertainment Inc.
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright
#  notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the distribution.
#  * Neither the name of the <organization> nor the
#  names of its contributors may be used to endorse or promote products
#  derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
########################################################################################
# Taken from https://github.com/lighttransport/nanogi/blob/master/cmake/FindGLFW.cmake, modified to fix naming issues
########################################################################################
# Find GLFW
#
# Try to find GLFW library.
# This module defines the following variables:
# - GLFW3_INCLUDE_DIRS
# - GLFW3_LIBRARIES
# - GLFW3_FOUND
#
# The following variables can be set as arguments for the module.
# - GLFW3_ROOT_DIR : Root library directory of GLFW
# - GLFW3_USE_STATIC_LIBS : Specifies to use static version of GLFW library (Windows only)
#
# References:
# - https://github.com/progschj/OpenGL-Examples/blob/master/cmake_modules/FindGLFW.cmake
# - https://bitbucket.org/Ident8/cegui-mk2-opengl3-renderer/src/befd47200265/cmake/FindGLFW.cmake
#

# Additional modules
include(FindPackageHandleStandardArgs)

if (WIN32)
    # Find include files
    find_path(
            GLFW3_INCLUDE_DIR
            NAMES GLFW/glfw3.h
            PATHS
            $ENV{PROGRAMFILES}/include
            ${GLFW_ROOT_DIR}/include
            DOC "The directory where GLFW/glfw3.h resides")

    # Use glfw3.lib for static library
    if (GLFW_USE_STATIC_LIBS)
        set(GLFW3_LIBRARY_NAME glfw3)
    else()
        set(GLFW3_LIBRARY_NAME glfw3dll)
    endif()

    # Find library files
    find_library(
            GLFW3_LIBRARY
            NAMES ${GLFW3_LIBRARY_NAME}
            PATHS
            $ENV{PROGRAMFILES}/lib
            ${GLFW_ROOT_DIR}/lib)

    unset(GLFW3_LIBRARY_NAME)
else()
    # Find include files
    find_path(
            GLFW3_INCLUDE_DIR
            NAMES GLFW/glfw3.h
            PATHS
            /usr/include
            /usr/local/include
            /sw/include
            /opt/local/include
            DOC "The directory where GLFW/glfw3.h resides")

    # Find library files
    # Try to use static libraries
    find_library(
            GLFW3_LIBRARY
            NAMES glfw3
            PATHS
            /usr/lib64
            /usr/lib
            /usr/local/lib64
            /usr/local/lib
            /sw/lib
            /opt/local/lib
            ${GLFW_ROOT_DIR}/lib
            DOC "The GLFW library")
endif()

# Handle REQUIRD argument, define *_FOUND variable
find_package_handle_standard_args(GLFW3 DEFAULT_MSG GLFW3_INCLUDE_DIR GLFW3_LIBRARY)

# Define GLFW_LIBRARIES and GLFW3_INCLUDE_DIRS
if (GLFW_FOUND)
    set(GLFW_LIBRARIES ${OPENGL_LIBRARIES} ${GLFW3_LIBRARY})
    set(GLFW3_INCLUDE_DIRS ${GLFW3_INCLUDE_DIR})
endif()

# Hide some variables
mark_as_advanced(GLFW3_INCLUDE_DIR GLFW3_LIBRARY)