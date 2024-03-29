#
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

include_directories(${SAMPLES_INCLUDE_DIR})
include_directories(${IL_INCLUDE_DIR})

# See top level CMakeLists.txt file for documentation of OPTIX_add_sample_executable.
OPTIX_add_sample_executable( OptaGen
	OptaGen.cpp
	sceneLoader.cpp
	Picture.cpp
	Texture.cpp
	sceneLoader.h
	material_parameters.h
	properties.h
	light_parameters.h
	state.h
	prd.h
	path.h
	rt_function.h
	MyAssert.h
	Picture.h
	Texture.h
	
	path_trace_camera.cu
	quad_intersect.cu
	disney.cu
	roughdielectric.cu
	lambert.cu
	glass.cu
	hit_program.cu
	light_hit_program.cu
	light_sample.cu
	background.cu
	triangle_mesh.cu
	sphere_intersect.cu

    # common headers
    ${SAMPLES_INCLUDE_DIR}/commonStructs.h
    ${SAMPLES_INCLUDE_DIR}/helpers.h
    ${SAMPLES_INCLUDE_DIR}/random.h
    )

