#pragma once

#include <optixu/optixu_vector_types.h>
#include "configs.h"

enum InteractionType
{
	DIFF, // sampled from the diffuse lobe
	GLOS, // sampled from 
	SPEC,
	REFL, // dielectric reflection
	TRAN  // dielectric transmission (refraction)
};

struct SampleRecord
{
	/* SMBC features */	
	// local features
	float subpixel_x, subpixel_y;
	float lens_u, lens_v;
	optix::float3 radiance;
	optix::float3 radiance_diffuse;
	optix::float3 albedo_at_first; // at the first geometric bounce
	optix::float3 albedo; // at the first diffuse bounce
	optix::float3 normal_at_first; // at the first geometric bounce
	optix::float3 normal; // at the first diffuse bounce
	float depth_at_first; // at the first geometric bounce
	float depth; // at the first diffuse bounce
	float visibility;
	float hasHit;

	// per-vertex features
	float probabilities[(MAX_DEPTH + 1) * 4];
	float light_directions[(MAX_DEPTH + 1) * 2];
	float bounce_types[(MAX_DEPTH + 1)];

	/* LLPM path descriptors */
	// local features
	float path_weight;
	float radiance_wo_weight;
	optix::float3 light_intensity;

	// per-vertex features
	optix::float3 throughputs[(MAX_DEPTH + 1)];
	float roughnesses[(MAX_DEPTH + 1)];
};