/* 
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived 
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "prd.h"

using namespace optix;

rtDeclareVariable(float3, background_light, , ); // horizon color
rtDeclareVariable(float3, background_dark, , );  // zenith color
rtDeclareVariable(float3, up, , );               // global up vector
rtDeclareVariable(int, option, , );				 // 1 if the envmap is given, 0 otherwise

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtTextureSampler<float4, 2> envmap;

// -----------------------------------------------------------------------------

/*RT_PROGRAM void miss()
{
  const float t = max(dot(ray.direction, up), 0.0f);
  const float3 result = lerp(background_light, background_dark, t);

  prd_radiance.radiance = result;
  prd_radiance.done = true;
}

inline __device__ float3 missColor(const optix::Ray &ray)
{
	const float3 unit_direction = normalize(ray.direction);
	const float t = 0.5f * (unit_direction.y + 1.0f);
	const float3 c = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
	return c;
}*/

RT_PROGRAM void miss()
{
	//const float t = max(dot(ray.direction, up), 0.0f);
	//const float3 result = lerp(background_light, background_dark, t);

	//prd_radiance.radiance = 0.01 * result;
	//prd_radiance.done = true;

	if (option == 0)
	{
		prd_radiance.done = true;
	}
	else
	{
		float theta = atan2f(ray.direction.x, ray.direction.z);
		float phi = M_PIf * 0.5f - acosf(ray.direction.y);
		float u = (theta + M_PIf) * (0.5f * M_1_PIf);
		float v = 0.5f * (1.0f + sin(phi));
		prd_radiance.radiance = prd_radiance.throughput * make_float3(tex2D(envmap, u, v)); // env map support
		prd_radiance.done = true;
	}
}