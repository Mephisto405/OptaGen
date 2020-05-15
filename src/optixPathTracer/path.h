#pragma once

#include <optixu/optixu_vector_types.h>

struct pathFeatures6
{
	optix::float3 rad[6];
	optix::float3 alb[6];
	optix::float3 nor[6];
};

//
struct path_aux_feat6
{
	optix::float3 throughput[6]; // estimate에서 p(x)와 L(.)를 제외한 나머지?
	int tag[6]; // BrdfType or diffuse/reflection/ 등 광현상에 관한 tag
	float roughness[6]; // roughness of brdf

	// Manual padding to float4 alignment.
	// CUDA L1 cache - 32 bytes unit
	// CUDA L2 cache - 128 bytes unit
	float unsused0;
	float unsused1;
};

//
struct sub_path_feat
{

};