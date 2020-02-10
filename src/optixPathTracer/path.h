#pragma once

#include <optixu/optixu_vector_types.h>

struct pathFeatures6
{
	optix::float3 rad[6];
	optix::float3 alb[6];
	optix::float3 nor[6];
};