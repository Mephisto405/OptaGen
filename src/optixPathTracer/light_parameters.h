#pragma once

#ifndef LIGHT_PARAMETER_H
#define LIGHT_PARAMETER_H

#include"rt_function.h"

enum LightType
{
	SPHERE, QUAD
};

/*
QUAD:
	position: world coordinates of the bottom-left vertex
	u: world coordinates of the top-left vertex
	v: world coordinates of the bottom-right vertex
SPHERE:
	position: world coordinates of the center
	radius: radius
*/
struct LightParameter
{
	optix::float3 position;
	optix::float3 normal;
	optix::float3 emission;
	optix::float3 u;
	optix::float3 v;
	float area;
	float radius;
	LightType lightType;
};

struct LightSample
{
	optix::float3 surfacePos;
	optix::float3 normal;
	optix::float3 emission;
	float pdf;
	
};

#endif
