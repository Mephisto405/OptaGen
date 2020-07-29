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
#include "helpers.h"
#include "prd.h"
#include "path.h"
#include "rt_function.h"
#include "random.h"
#include <assert.h>
#include <stdio.h>

using namespace optix;


rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float3, bad_color, , );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(float3, cutoff_color, , );
rtDeclareVariable(int, max_depth, , );
rtBuffer<float4, 2>              output_buffer;
rtBuffer<PathFeature[4], 2>      mbpf_buffer; /* Multiple-bounced feature buffer */
rtBuffer<float4, 2>              accum_buffer;
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(unsigned int, frame, , );
rtDeclareVariable(unsigned int, curr_time, , );
rtDeclareVariable(int, mbpf_frames, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );


__device__ inline float4 ToneMap(const float4& c, float limit)
{
	float luminance = 0.3f*c.x + 0.6f*c.y + 0.1f*c.z;

	float4 col = c * 1.0f / (1.0f + luminance / limit);
	return make_float4(col.x, col.y, col.z, 1.0f);
}

__device__ inline float4 LinearToSrgb(const float4& c)
{
	const float kInvGamma = 1.0f / 2.2f;
	return make_float4(powf(c.x, kInvGamma), powf(c.y, kInvGamma), powf(c.z, kInvGamma), c.w);
}

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

__device__ inline float3 clip(const float3& c)
{
	return make_float3(
		c.x < 0 ? 0 : c.x > 1.0 ? 1.0 : c.x,
		c.y < 0 ? 0 : c.y > 1.0 ? 1.0 : c.y,
		c.z < 0 ? 0 : c.z > 1.0 ? 1.0 : c.z
		);
}

RT_PROGRAM void pinhole_camera()
{
	size_t2 screen = output_buffer.size();
	unsigned int seed = tea<16>(screen.x*launch_index.y + launch_index.x, frame + curr_time);

	// Subpixel jitter: send the ray through a different position inside the pixel each time,
	// to provide antialiasing.
	float2 subpixel_jitter = frame == 0 ? make_float2(0.0f) : make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

	float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x*U + d.y*V + W);

	PerRayData_radiance prd;
	prd.depth = 0;
	prd.seed = seed;
	prd.done = false;
	prd.pdf = 0.0f;
	prd.specularBounce = false;
	prd.thpt_at_vtx = make_float3(0.0f);
	prd.tag = DIFF;
	prd.roughness = 0.0f;


	// These represent the current shading state and will be set by the closest-hit or miss program

	// attenuation (<= 1) from surface interaction.
	prd.throughput = make_float3(1.0f);

	// light from a light source or miss program
	prd.radiance = make_float3(0.0f);

	// next ray to be traced
	prd.origin = make_float3(0.0f);
	prd.bsdfDir = make_float3(0.0f);

	float3 result = make_float3(0.0f);

	PathFeature pf{
		{ optix::make_float3(0.f) }, { DIFF }, { 0.0f }, // multi-bounce features
		make_float3(0.f), make_float3(0.f), make_float3(0.f), // first-bounce features
		1.0f, // MC probability
	};

	// Main render loop. This is not recursive, and for high ray depths
	// will generally perform better than tracing radiance rays recursively
	// in closest hit programs.
	for (;;) {
		optix::Ray ray(ray_origin, ray_direction, /*ray type*/ 0, scene_epsilon);
		prd.wo = -ray.direction;
		rtTrace(top_object, ray, prd);

		if (prd.depth == 0)
		{
			pf.albedo = clip(prd.albedo);
			pf.normal = (prd.normal.x == 0.f && prd.normal.y == 0.f && prd.normal.z == 0.f) ?
				prd.normal :
				0.5f * normalize(prd.normal) + 0.5f;
		}

		if (prd.done)
			break;

		/* Path features */
		pf.prob *= prd.pdf;
		if (prd.depth < 6)
		{
			pf.throughput[prd.depth] = prd.thpt_at_vtx;
			pf.tag[prd.depth] = (float)prd.tag;
			pf.roughness[prd.depth] = prd.roughness;
		}
		else
		{
			pf.throughput[5] *= prd.thpt_at_vtx;
		}

		if (prd.done || prd.depth >= max_depth)
			break;

		prd.depth++;

		// Update ray data for the next path segment
		ray_origin = prd.origin;
		ray_direction = prd.bsdfDir;
	}

	pf.radiance = prd.radiance;
	result = prd.radiance;

	float4 acc_val = accum_buffer[launch_index];
	if (frame > 0) {
		acc_val = lerp(acc_val, make_float4(result, 0.f), 1.0f / static_cast<float>(frame + 1));
	}
	else {
		acc_val = make_float4(result, 0.f);
	}

	//float4 val = LinearToSrgb(ToneMap(acc_val, 1.5));
	//float4 val = LinearToSrgb(acc_val);

	output_buffer[launch_index] = acc_val; // uint
	accum_buffer[launch_index] = acc_val;
	if (frame < mbpf_frames)
		mbpf_buffer[launch_index][frame] = pf;
}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf("Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y);
	output_buffer[launch_index] = make_float4(bad_color);
}