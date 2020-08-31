#pragma once

#include <optixu/optixu_vector_types.h>
#include "configs.h"

/*
For each BxDF, the flags should have at least one of BSDF_REFLECTION 
or BSDF_TRANSMISSION set and exactly one of the diffuse, glossy, and specular flags.
*/
enum BxDFType {
	BSDF_REFLECTION = 1 << 0,
	BSDF_TRANSMISSION = 1 << 1,
	BSDF_DIFFUSE = 1 << 2,
	BSDF_GLOSSY = 1 << 3,
	BSDF_SPECULAR = 1 << 4,
	BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR |	BSDF_REFLECTION | BSDF_TRANSMISSION,
};

struct SampleRecord
{
	/* SMBC features */	
	// local features
	float subpixel_x, subpixel_y;
	optix::float3 radiance;
	optix::float3 radiance_diffuse; /**/
	optix::float3 albedo_at_first; // at the first geometric bounce
	optix::float3 albedo; // at the first diffuse bounce
	optix::float3 normal_at_first; // at the first geometric bounce
	optix::float3 normal; // at the first diffuse bounce
	float depth_at_first; // at the first geometric bounce
	float depth; // at the first diffuse bounce
	float visibility; /**/
	float hasHit;

	// per-vertex features
	float probabilities[(MAX_DEPTH + 1) * 4]; /**/
	float light_directions[(MAX_DEPTH + 1) * 2]; /**/
	float bounce_types[(MAX_DEPTH + 1)];


	/* KPCN features */
	optix::float3 albedo_at_diff;
	optix::float3 normal_at_diff;
	float depth_at_diff;


	/* LLPM path descriptors */
	// local features
	float path_weight; /**/
	optix::float3 radiance_wo_weight; /**/
	optix::float3 light_intensity;

	// per-vertex features
	optix::float3 throughputs[(MAX_DEPTH + 1)]; /**/
	float roughnesses[(MAX_DEPTH + 1)]; /**/
};