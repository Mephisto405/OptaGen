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
#include "configs.h"

using namespace optix;

rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, );
rtDeclareVariable(float3, back_hit_point, attribute back_hit_point, );
rtDeclareVariable(float3, texcoord, attribute texcoord, );

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(int, max_depth, , );

rtBuffer< rtCallableProgramId<void(MaterialParameter &mat, State &state, PerRayData_radiance &prd)> > sysBRDFPdf;
rtBuffer< rtCallableProgramId<void(MaterialParameter &mat, State &state, PerRayData_radiance &prd)> > sysBRDFSample;
rtBuffer< rtCallableProgramId<float3(MaterialParameter &mat, State &state, PerRayData_radiance &prd)> > sysBRDFEval;
rtBuffer< rtCallableProgramId<void(const LightParameter &light, const float3 &surfacePos, unsigned int &seed, LightSample &lightSample)> > sysLightSample;

rtBuffer<MaterialParameter> sysMaterialParameters;
rtDeclareVariable(int, materialId, , );
rtDeclareVariable(int, sysNumberOfLights, , );

rtBuffer<LightParameter> sysLightParameters;


RT_FUNCTION float3 DirectLight(MaterialParameter &mat, State &state)
{
	float3 L = make_float3(0.0f);

	//Pick a light to sample
	int index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * sysNumberOfLights)), 0, sysNumberOfLights - 1);
	LightParameter light = sysLightParameters[index];
	LightSample lightSample;

	float3 surfacePos = state.fhp;

	sysLightSample[light.lightType](light, surfacePos, prd.seed, lightSample);

	if (0.0f < lightSample.pdf)
	{
		prd.bsdfDir = lightSample.direction;
		sysBRDFPdf[mat.brdf](mat, state, prd);
		float3 f = sysBRDFEval[mat.brdf](mat, state, prd);

		if (0.0f < prd.pdf && (f.x != 0.0f || f.y != 0.0f || f.z != 0.0f))
		{
			PerRayData_shadow prdShadow;
			prdShadow.inShadow = false;

			Ray shadowRay = make_Ray(surfacePos, lightSample.direction, 1, scene_epsilon, lightSample.distance - scene_epsilon);
			rtTrace(top_object, shadowRay, prdShadow);

			prd.inShadow = prdShadow.inShadow;

			if (!prdShadow.inShadow)
			{
				const float misWeight = powerHeuristic(lightSample.pdf, prd.pdf);
				L = misWeight * prd.throughput * f * lightSample.emission / max(1e-3f, lightSample.pdf);
			}
		}
		else
			prd.inShadow = true;
	}
	else
		prd.inShadow = true;

	return L;
}


RT_PROGRAM void closest_hit()
{
	const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	const float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	MaterialParameter mat = sysMaterialParameters[materialId];

	if (mat.albedoID != RT_TEXTURE_ID_NULL)
	{
		const float3 texColor = make_float3(optix::rtTex2D<float4>(mat.albedoID, texcoord.x, texcoord.y));
		mat.color = make_float3(powf(texColor.x, 2.2f), powf(texColor.y, 2.2f), powf(texColor.z, 2.2f));
	}

	State state;
	state.fhp = front_hit_point;
	state.bhp = back_hit_point;
	state.normal = world_shading_normal;
	state.ffnormal = ffnormal;
	prd.wo = -ray.direction;
	prd.ray_dist += sqrtf(dot(state.fhp - ray.origin, state.fhp - ray.origin));

	// Emissive radiance
	prd.radiance += mat.emission * prd.throughput;

	// TODO: Clean up handling of specular bounces
	prd.specularBounce = mat.brdf == GLASS || mat.brdf == ROUGHDIELECTRIC ? true : false;
	bool bsdf_has_diffuse = (mat.brdf == LAMBERT || (mat.brdf == DISNEY && mat.metallic != 1.0f));
	bool bsdf_has_nonspecular = bsdf_has_diffuse || (mat.brdf == ROUGHDIELECTRIC);

	prd.is_first_non_specular = false;
	if (!prd.found_non_specular && bsdf_has_nonspecular)
	{
		prd.is_first_non_specular = true;
		prd.found_non_specular = true;
	}

	// Direct light Sampling
	optix::float3 direct_radiance = DirectLight(mat, state);
	if (!prd.specularBounce && prd.depth < MAX_DEPTH)
		prd.radiance += direct_radiance;

	// BRDF Sampling
	sysBRDFSample[mat.brdf](mat, state, prd);
	sysBRDFPdf[mat.brdf](mat, state, prd);
	float3 f = sysBRDFEval[mat.brdf](mat, state, prd);

	prd.albedo = mat.color; 
	prd.normal = normalize(rtTransformNormal(RT_WORLD_TO_OBJECT, ffnormal)); // camera-space normal
	prd.hasHit = true;

	if (prd.pdf > 0.0f)
	{
		prd.throughput *= f / prd.pdf;
	}
	else
	{
		prd.done = true;
	}
}


RT_PROGRAM void any_hit()
{
	prd_shadow.inShadow = true;
	rtTerminateRay();
}
