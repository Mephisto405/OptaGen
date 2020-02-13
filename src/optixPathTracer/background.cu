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
#include "helpers.h"
#include "light_parameters.h"
#include <assert.h>

using namespace optix;

rtDeclareVariable(float3, background_light, , ); // horizon color
rtDeclareVariable(float3, background_dark, , );  // zenith color
rtDeclareVariable(float3, up, , );               // global up vector
rtDeclareVariable(int, option, , );				 // 1 if the envmap is given, 0 otherwise

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );

rtBuffer<LightParameter> sysLightParameters;


RT_PROGRAM void miss()
{
	if (option == 0)
	{
		prd.done = true;
	}
	else
	{		
		const LightParameter light = sysLightParameters[0];

		float3 dir = normalize(ray.direction); // might be unnecessary
		float theta = acosf(-dir.y); // theta == 0.0f is south pole, theta == M_PIf is north pole
		float phi = (dir.x == 0.0f && dir.z == 0.0f) ? 0.0f : atan2f(dir.x, -dir.z); // Starting on positive z-axis going around clockwise (to negative x-axis)
		float u = (M_PI + phi) * (0.5f * M_1_PIf) /* + 0.5f */;
		float v = theta * M_1_PIf;

		const float3 emission = make_float3(optix::rtTex2D<float4>(light.idEnvironmentTexture, u, v)); // env map support

		float misWeight = 1.0f;
		if (!prd.specularBounce && prd.depth != 0)
		{
			//assert(sysLightParameters[0].lightType != LightType::ENVMAP);
			const float pdfLight = 0.3333333333f * (emission.x + emission.y + emission.z) / light.environmentIntegral;
			misWeight = powerHeuristic(prd.pdf, pdfLight);
		}

		prd.radiance = misWeight * prd.throughput * emission;
		prd.done = true;
	}
}