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
#include "samplerecord.h"
#include "configs.h"
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
rtDeclareVariable(float, scene_radius, , );
rtBuffer<float4, 2>              output_buffer;
rtBuffer<SampleRecord[MAX_SAMPLES], 2>      mbpf_buffer; /* Multiple-bounced feature buffer */
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
	/* Sub-pixel jittering */
	size_t2 screen = output_buffer.size();
	unsigned int seed = tea<16>(screen.x*launch_index.y + launch_index.x, frame + curr_time);

	float subpixel_x = frame == 0 ? 0.5f : rnd(seed);
	float subpixel_y = frame == 0 ? 0.5f : rnd(seed);
	float2 subpixel_jitter = make_float2(subpixel_x - 0.5f, subpixel_y - 0.5f);
	float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x*U + d.y*V + W);


	/* Records */
	// ray records
	PerRayData_radiance prd = {};
	prd.seed = seed;
	prd.throughput = make_float3(1.0f);
	prd.throughput_diffuse = make_float3(1.0f);

	// sample records
	SampleRecord sr = {};
	sr.subpixel_x = subpixel_x;
	sr.subpixel_y = subpixel_y;
	float depth_norm = scene_radius > 0.0f ? 1.0f / (10.0f * scene_radius) : 1.0f;


	/* Main rendering loop */
	float3 result = make_float3(0.0f);
	for (;;) {
		optix::Ray ray(ray_origin, ray_direction, 0 /*ray type*/, scene_epsilon);
		prd.wo = -ray.direction;
		rtTrace(top_object, ray, prd);


		/* post-processing */
		// at the first geometric bounce
		if (prd.depth == 0)
		{
			sr.albedo_at_first = prd.albedo;
			sr.normal_at_first = prd.normal;
			sr.depth_at_first = prd.hasHit ? prd.ray_dist * depth_norm : -0.1f;
			sr.visibility = prd.hasHit ? (!prd.inShadow ? 1.0f : 0.0f) : 0.0f;
			sr.hasHit = prd.hasHit ? 1.0f : 0.0f;
		}
		
		// TODO(iycho): dirty code and not work properly
		if (prd.depth == 1 && !prd.hasHit && dot(prd.light_intensity, prd.light_intensity) != 0)
		{
			// the object is visible if the ray hit a non-black light at the second bounce
			sr.visibility = true;
		}

		// either at the first non specular bounce 
		// or no specular bounce until the end of light transport
		if (prd.is_first_non_specular || (sr.depth == 0.0f && !prd.hasHit))
		{
			sr.albedo = prd.albedo;
			sr.normal = prd.normal;
			sr.depth = prd.hasHit ? prd.ray_dist * depth_norm : -0.1f;
		}

		if (!prd.hasHit)
		{
			sr.light_intensity = prd.light_intensity;
		}
		else
		{
			if (prd.depth == 0)
			{
				sr.path_weight = 1.0f;
			}
			sr.path_weight *= prd.pdf;
			sr.radiance_wo_weight *= prd.thpt_at_vtx;

			// record sample data
			sr.throughputs[prd.depth] = prd.thpt_at_vtx;
			sr.bounce_types[prd.depth] = (float)prd.bounce_type;
			sr.roughnesses[prd.depth] = prd.roughness;
		}

		/* exit if light cannot transfer further */
		if (prd.done || prd.depth >= MAX_DEPTH) // >= max_depth
			break;

		prd.depth++;

		// update ray data for the next path segment
		ray_origin = prd.origin;
		ray_direction = prd.bsdfDir;
	}

	sr.radiance = prd.radiance;
	sr.radiance_diffuse = prd.radiance_diffuse;
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
		mbpf_buffer[launch_index][frame] = sr;
}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf("Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y);
	output_buffer[launch_index] = make_float4(bad_color);
}