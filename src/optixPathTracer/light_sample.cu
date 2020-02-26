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
#include <optixu_matrix_namespace.h>
#include "helpers.h"
#include "prd.h"
#include "random.h"
#include "rt_function.h"
#include "material_parameters.h"
#include "light_parameters.h"
#include "state.h"
#include <assert.h>
#include <stdio.h>

using namespace optix;

rtDeclareVariable(int, sysNumberOfLights, , );

rtBuffer<LightParameter> sysLightParameters;

RT_FUNCTION float3 UniformSampleSphere(float u1, float u2)
{
	float z = 1.f - 2.f * u1;
	float r = sqrtf(max(0.f, 1.f - z * z));
	float phi = 2.f * M_PIf * u2;
	float x = r * cosf(phi);
	float y = r * sinf(phi);

	return make_float3(x, y, z);
}


RT_CALLABLE_PROGRAM void envmap_sample(const LightParameter &light, const float3 &surfacePos, unsigned int &seed, LightSample &lightSample)
{
	const float r1 = rnd(seed);
	const float r2 = rnd(seed);

	const unsigned int sizeU = static_cast<unsigned int>(light.idEnvironmentCDF_U.size().x);
	const unsigned int sizeV = static_cast<unsigned int>(light.idEnvironmentCDF_V.size());

	unsigned int ilo = 0; // lower limit
	unsigned int ihi = sizeV - 1; // higher limit

	//printf("(GPU) type: %d", light.lightType);
	//printf("(GPU) textureID: %d\n", light.idEnvironmentTexture);
	//printf("(GPU) Envmap integral: %f\n", sysLightParameters[0].environmentIntegral);

	while (ilo != ihi - 1)
	{
		const unsigned int i = (ilo + ihi) >> 1;
		//printf("%u %u\n", light.idEnvironmentCDF_U.size().x, light.idEnvironmentCDF_U.size().y);
		const float cdf = light.idEnvironmentCDF_V[i];
		//assert(LightType::ENVMAP != light.lightType);
		if (r2 < cdf) // If the cdf is greater than the sample, use that as new higher limit.
		{
			ihi = i;
		}
		else // If the sample is greater than or equal to the CDF value, use that as new lower limit.
		{
			ilo = i;
		}
	}
	
	uint2 index;
	index.y = ilo;

	ilo = 0;
	ihi = sizeU - 1;

	while (ilo != ihi - 1)
	{
		index.x = (ilo + ihi) >> 1;
		const float cdf = light.idEnvironmentCDF_U[index];
		if (r1 < cdf) // If the cdf is greater than the sample, use that as new higher limit.
		{
			ihi = index.x;
		}
		else // If the sample is greater than or equal to the CDF value, use that as new lower limit.
		{
			ilo = index.x;
		}
	}
	
	index.x = ilo;
	
	// Continuous sampling of the CDF.
	// Continuous sampling of the CDF.
	const float cdfLowerU = light.idEnvironmentCDF_U[index];
	const float cdfUpperU = light.idEnvironmentCDF_U[make_uint2(index.x + 1, index.y)];
	const float du = (r1- cdfLowerU) / (cdfUpperU - cdfLowerU);

	const float cdfLowerV = light.idEnvironmentCDF_V[index.y];
	const float cdfUpperV = light.idEnvironmentCDF_V[index.y + 1];
	const float dv = (r2 - cdfLowerV) / (cdfUpperV - cdfLowerV);

	// Texture lookup coordinates.
	const float u = (float(index.x) + du) / float(sizeU - 1);
	const float v = (float(index.y) + dv) / float(sizeV - 1);

	// Light sample direction vector polar coordinates. This is where the environment rotation happens!
	// DAR FIXME Use a light.matrix to rotate the resulting vector instead.
	const float phi = (u /* - 0.5f */) * 2.0f * M_PIf;
	const float theta = v * M_PIf; // theta == 0.0f is south pole, theta == M_PIf is north pole.

	const float sinTheta = sinf(theta);
	// The miss program places the 1->0 seam at the positive z-axis and looks from the inside.
	lightSample.direction = make_float3(-sinf(phi) * sinTheta,  // Starting on positive z-axis going around clockwise (to negative x-axis).
		-cosf(theta),           // From south pole to north pole.
		cosf(phi) * sinTheta); // Starting on positive z-axis.

	// Note that environment lights do not set the light sample position!
	lightSample.distance = RT_DEFAULT_MAX; // Environment light.

	const float3 emission = make_float3(optix::rtTex2D<float4>(light.idEnvironmentTexture, u, v));
	// Explicit light sample. The returned emission must be scaled by the inverse probability to select this light.
	lightSample.emission = emission * float(sysNumberOfLights);
	// For simplicity we pretend that we perfectly importance-sampled the actual texture-filtered environment map
	// and not the Gaussian-smoothed one used to actually generate the CDFs and uniform sampling in the texel.
	lightSample.pdf = 0.3333333333f * (emission.x + emission.y + emission.z) / light.environmentIntegral;
}


RT_CALLABLE_PROGRAM void sphere_sample(const LightParameter &light, const float3 &surfacePos, unsigned int &seed, LightSample &lightSample)
{
	const float r1 = rnd(seed);
	const float r2 = rnd(seed);
	
	lightSample.pdf = 0.0f;

	optix::float3 lightSamplePos = light.position + UniformSampleSphere(r1, r2) * light.radius;
	lightSample.direction = lightSamplePos - surfacePos;
	lightSample.distance = length(lightSample.direction);

	if (1.0e-6f < lightSample.distance)
	{
		lightSample.direction /= lightSample.distance;
		optix::float3 lightNormal = normalize(lightSamplePos - light.position);

		const float cosTheta = dot(-lightSample.direction, lightNormal); // light이 surface를 바라보는 방향이어야 한다.
		if (1.0e-6f < cosTheta)
		{
			lightSample.emission = light.emission * float(sysNumberOfLights);
			lightSample.pdf = (lightSample.distance * lightSample.distance) / (light.area * cosTheta);
		}
	}
}


/*
 lightSample의 pdf, distance, direction, emission을 구한다.
 pdf: Monte Carlo estimator와 power heuristic 계수를 계산하기 위해
 distance: 빛이 닿을 수 있는 valid한 방향인지 확인하기 위해 shadowRay를 쏠 방향
 direction: shadowRay를 쏠 거리
 emission: emissive radiance
 */
RT_CALLABLE_PROGRAM void quad_sample(const LightParameter &light, const float3 &surfacePos, unsigned int &seed, LightSample &lightSample)
{
	const float r1 = rnd(seed);
	const float r2 = rnd(seed);

	lightSample.pdf = 0.0f;

	optix::float3 lightSamplePos = light.position + light.u * r1 + light.v * r2;
	lightSample.direction = lightSamplePos - surfacePos;
	lightSample.distance = length(lightSample.direction);

	if (1.0e-6f < lightSample.distance)
	{
		lightSample.direction /= lightSample.distance;

		const float cosTheta = dot(-lightSample.direction, light.normal); // light이 surface를 바라보는 방향이어야 한다.
		if (1.0e-6f < cosTheta)
		{
			lightSample.emission = light.emission * float(sysNumberOfLights);
			lightSample.pdf = (lightSample.distance * lightSample.distance) / (light.area * cosTheta);
		}
	}
}
