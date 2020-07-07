#pragma once

#include <optixu/optixu_vector_types.h>

enum InteractionType
{
	DIFF, // sampled from the diffuse lobe
	GLOS, // sampled from 
	SPEC,
	REFL, // dielectric reflection
	TRAN  // dielectric transmission (refraction)
};

struct pathFeatures6
{
	optix::float3 rad[6];
	optix::float3 alb[6];
	optix::float3 nor[6];
};

// multi-bounce path feature
struct PathFeature
{
	// for manifold learning phase
	optix::float3 throughput[6];
	float tag[6];
	float roughness[6];

	// for denoising phase
	optix::float3 radiance;
	optix::float3 albedo;
	optix::float3 normal;

	// MC probability
	float prob;
};

//
struct sub_path_feat
{

};