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
#include "light_parameters.h"
#include "state.h"

using namespace optix;

rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, );
rtDeclareVariable(float3, back_hit_point, attribute back_hit_point, );

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, hit_dist, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(float, scene_epsilon, , );

rtBuffer<LightParameter> sysLightParameters;
rtDeclareVariable(int, lightMaterialId, , );


__device__ inline float3 ToneMap(const float3& c, float limit)
{
	float luminance = 0.3f*c.x + 0.6f*c.y + 0.1f*c.z;

	float3 col = c * 1.0f / (1.0f + luminance / limit);
	return make_float3(col.x, col.y, col.z);
}

__device__ inline float3 LinearToSrgb(const float3& c)
{
	const float kInvGamma = 1.0f / 2.2f;
	return make_float3(powf(c.x, kInvGamma), powf(c.y, kInvGamma), powf(c.z, kInvGamma));
}


RT_PROGRAM void closest_hit()
{
	const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	const float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	LightParameter light = sysLightParameters[lightMaterialId];
	float cosTheta;
	if (light.lightType == QUAD)
	{
		cosTheta = dot(-ray.direction, light.normal);
	}
	else if (light.lightType == SPHERE)
	{
		const float3 lightNormal = normalize(front_hit_point - light.position);
		cosTheta = dot(-ray.direction, lightNormal);
	}


	if ((light.lightType == QUAD || light.lightType == SPHERE) && cosTheta > 0.0f)
	{
		if (prd.depth == 0 || prd.specularBounce)
		{
			prd.radiance += light.emission * prd.throughput;
		}
		else
		{
			float lightPdf = (hit_dist * hit_dist) / (light.area * clamp(cosTheta, 1.e-3f, 1.0f));
			prd.radiance += powerHeuristic(prd.pdf, lightPdf) * light.emission * prd.throughput;
		}

		prd.albedo = LinearToSrgb(ToneMap(light.emission, 1.5));
		prd.normal = ffnormal;
	}
	else
	{
		prd.albedo = make_float3(0.f);
		prd.normal = ffnormal;
	}

	prd.done = true;
}