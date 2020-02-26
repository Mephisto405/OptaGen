#pragma once

#ifndef LIGHT_PARAMETER_H
#define LIGHT_PARAMETER_H

#include"rt_function.h"

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

enum LightType
{
	ENVMAP, SPHERE, QUAD
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
	LightType lightType;

	optix::float3 position;
	optix::float3 u;
	optix::float3 v;
	optix::float3 normal;
	float area;
	float radius;
	optix::float3 emission;

	// Bindless texture and buffer IDs. Only valid for spherical environment lights.
	int                  idEnvironmentTexture;
	rtBufferId<float, 2> idEnvironmentCDF_U;   // rtBufferId fields are integers.
	rtBufferId<float, 1> idEnvironmentCDF_V;
	float                environmentIntegral;
	
	// Manual padding to float4 alignment.
	float unsused0;
	float unsused1;
};

struct LightSample
{
	//optix::float3 surfacePos;
	//optix::float3 normal;
	//optix::float3 emission;
	//float pdf;
	
	//optix::float3 surfacePos;
	//optix::float3 normal;
	//int           index;
	optix::float3 direction;
	float         distance;
	optix::float3 emission;
	float         pdf;
};

#endif
