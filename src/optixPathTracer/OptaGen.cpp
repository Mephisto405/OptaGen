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

//-----------------------------------------------------------------------------
//
// optixPathTracer: A path tracer using the disney brdf.
//
//-----------------------------------------------------------------------------

#ifndef __APPLE__
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#  endif
#endif

#include <GLFW/glfw3.h>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <sutil.h>
#include <ctime>
#include "commonStructs.h"
#include "sceneLoader.h"
#include "light_parameters.h"
#include "material_parameters.h"
#include "properties.h"
#include "path.h"
#include <IL/il.h>
#include <Camera.h>
#include <OptiXMesh.h>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <dirent.h>
#include <stdint.h>

using namespace optix;

const char* const SAMPLE_NAME = "optixPathTracer";
std::string SAVE_DIR = "";

const int NUMBER_OF_BRDF_INDICES = 4;
const int NUMBER_OF_LIGHT_INDICES = 2;
optix::Buffer m_bufferBRDFSample;
optix::Buffer m_bufferBRDFEval;
optix::Buffer m_bufferBRDFPdf;

optix::Buffer m_bufferLightSample;
optix::Buffer m_bufferMaterialParameters;
optix::Buffer m_bufferLightParameters;

double elapsedTime = 0;
double lastTime = 0;

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------
Properties properties;
Context      context = 0;
Scene* scene;


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

static std::string ptxPath( const std::string& cuda_file )
{
    return
        std::string(sutil::samplesPTXDir()) +
        "/" + std::string(SAMPLE_NAME) + "_generated_" +
        cuda_file +
        ".ptx";
}

optix::GeometryInstance createSphere(optix::Context context,
	optix::Material material,
	float3 center,
	float radius)
{
	optix::Geometry sphere = context->createGeometry();
	sphere->setPrimitiveCount(1u);
	const std::string ptx_path = ptxPath("sphere_intersect.cu");
	sphere->setBoundingBoxProgram(context->createProgramFromPTXFile(ptx_path, "bounds"));
	sphere->setIntersectionProgram(context->createProgramFromPTXFile(ptx_path, "sphere_intersect_robust"));

	sphere["center"]->setFloat(center);
	sphere["radius"]->setFloat(radius);

	optix::GeometryInstance instance = context->createGeometryInstance(sphere, &material, &material + 1);
	return instance;
}

optix::GeometryInstance createQuad(optix::Context context,
	optix::Material material,
	float3 v1, float3 v2, float3 anchor, float3 n)
{
	optix::Geometry quad = context->createGeometry();
	quad->setPrimitiveCount(1u);
	const std::string ptx_path = ptxPath("quad_intersect.cu");
	quad->setBoundingBoxProgram(context->createProgramFromPTXFile(ptx_path, "bounds"));
	quad->setIntersectionProgram(context->createProgramFromPTXFile(ptx_path, "intersect"));

	float3 normal = normalize(cross(v1, v2));
	float4 plane = make_float4(normal, dot(normal, anchor));
	v1 *= 1.0f / dot(v1, v1);
	v2 *= 1.0f / dot(v2, v2);
	quad["v1"]->setFloat(v1);
	quad["v2"]->setFloat(v2);
	quad["anchor"]->setFloat(anchor);
	quad["plane"]->setFloat(plane);

	optix::GeometryInstance instance = context->createGeometryInstance(quad, &material, &material + 1);
	return instance;
}


static Buffer getOutputBuffer()
{
    return context[ "output_buffer" ]->getBuffer();
}

static Buffer getNormalBuffer()
{
	return context["normal_buffer"]->getBuffer();
}

static Buffer getMBFBuffer()
{
	return context["mbf_buffer"]->getBuffer();
}

void destroyContext()
{
    if( context )
    {
        context->destroy();
        context = 0;
    }
}


void createContext( bool use_pbo, unsigned int max_depth )
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );

    // Note: this sample does not need a big stack size even with high ray depths, 
    // because rays are not shot recursively.
    context->setStackSize( 800 );

    // Note: high max depth for reflection and refraction through glass
	context["max_depth"]->setInt( max_depth );
    context["cutoff_color"]->setFloat( 0.0f, 0.0f, 0.0f );
    context["frame"]->setUint( 0u );
    context["scene_epsilon"]->setFloat( 1.e-3f );

    Buffer buffer = sutil::createOutputBuffer( context, RT_FORMAT_UNSIGNED_BYTE4, 
		scene->properties.width, scene->properties.height, use_pbo );
    context["output_buffer"]->set( buffer );

	Buffer normal_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3,
		scene->properties.width, scene->properties.height);
	context["normal_buffer"]->set(normal_buffer);

	/* Multiple-bounced feature buffer */
	Buffer mbf_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER,
		scene->properties.width, scene->properties.height);
	mbf_buffer->setElementSize(sizeof(pathFeatures6)); // a user-defined type whose size is specified with *@ref rtBufferSetElementSize.
	context["mbf_buffer"]->set(mbf_buffer); 

    // Accumulation buffer
    Buffer accum_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
            RT_FORMAT_FLOAT4, scene->properties.width, scene->properties.height);
    context["accum_buffer"]->set( accum_buffer );

    // Ray generation program
    std::string ptx_path( ptxPath( "path_trace_camera.cu" ) );
    Program ray_gen_program = context->createProgramFromPTXFile( ptx_path, "pinhole_camera" );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXFile( ptx_path, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    ptx_path = ptxPath( "background.cu" );
    context->setMissProgram( 0, context->createProgramFromPTXFile( ptx_path, "miss" ) );
	const std::string texture_filename = scene->dir + scene->properties.bg_file_name;
	std::cerr << texture_filename << std::endl;
	context["background_light"]->setFloat(1.0f, 1.0f, 1.0f);
	context["background_dark"]->setFloat(0.3945f, 0.0f, 0.4945f);
	context["up"]->setFloat(0.0f, 1.0f, 0.0f); 
	context["option"]->setInt((int)(scene->properties.bg_file_name != ""));
	context["envmap"]->setTextureSampler(sutil::loadTexture(context, texture_filename, optix::make_float3(1.0f)));

	Program prg;
	// BRDF sampling functions.
	m_bufferBRDFSample = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_BRDF_INDICES);
	int* brdfSample = (int*) m_bufferBRDFSample->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	prg = context->createProgramFromPTXFile(ptxPath("disney.cu"), "Sample");
	brdfSample[BrdfType::DISNEY] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("glass.cu"), "Sample");
	brdfSample[BrdfType::GLASS] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("lambert.cu"), "Sample");
	brdfSample[BrdfType::LAMBERT] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("roughdielectric.cu"), "Sample");
	brdfSample[BrdfType::ROUGHDIELECTRIC] = prg->getId();
	m_bufferBRDFSample->unmap();
	context["sysBRDFSample"]->setBuffer(m_bufferBRDFSample);
	
	// BRDF Eval functions.
	m_bufferBRDFEval = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_BRDF_INDICES);
	int* brdfEval = (int*) m_bufferBRDFEval->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	prg = context->createProgramFromPTXFile(ptxPath("disney.cu"), "Eval");
	brdfEval[BrdfType::DISNEY] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("glass.cu"), "Eval");
	brdfEval[BrdfType::GLASS] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("lambert.cu"), "Eval");
	brdfEval[BrdfType::LAMBERT] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("roughdielectric.cu"), "Eval");
	brdfEval[BrdfType::ROUGHDIELECTRIC] = prg->getId();
	m_bufferBRDFEval->unmap();
	context["sysBRDFEval"]->setBuffer(m_bufferBRDFEval);
	
	// BRDF Pdf functions.
	m_bufferBRDFPdf = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_BRDF_INDICES);
	int* brdfPdf = (int*) m_bufferBRDFPdf->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	prg = context->createProgramFromPTXFile(ptxPath("disney.cu"), "Pdf");
	brdfPdf[BrdfType::DISNEY] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("glass.cu"), "Pdf");
	brdfPdf[BrdfType::GLASS] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("lambert.cu"), "Pdf");
	brdfPdf[BrdfType::LAMBERT] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("roughdielectric.cu"), "Pdf");
	brdfPdf[BrdfType::ROUGHDIELECTRIC] = prg->getId();
	m_bufferBRDFPdf->unmap();
	context["sysBRDFPdf"]->setBuffer(m_bufferBRDFPdf);

	// Light sampling functions.
	m_bufferLightSample = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_LIGHT_INDICES);
	int* lightsample = (int*)m_bufferLightSample->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	prg = context->createProgramFromPTXFile(ptxPath("light_sample.cu"), "sphere_sample");
	lightsample[LightType::SPHERE] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("light_sample.cu"), "quad_sample");
	lightsample[LightType::QUAD] = prg->getId();
	m_bufferLightSample->unmap();
	context["sysLightSample"]->setBuffer(m_bufferLightSample);

	// PostprocessingStage tonemapper = context->createBuiltinPostProcessingStage("TonemapperSimple");
	// http://on-demand.gputechconf.com/gtc/2018/presentation/s8518-an-introduction-to-optix.pdf
}

Material createMaterial(const MaterialParameter &mat, int index)
{
	const std::string ptx_path = ptxPath( "hit_program.cu" );
	Program ch_program = context->createProgramFromPTXFile( ptx_path, "closest_hit" );
	Program ah_program = context->createProgramFromPTXFile(ptx_path, "any_hit");
	
	Material material = context->createMaterial();
	material->setClosestHitProgram( 0, ch_program );
	material->setAnyHitProgram(1, ah_program);
	
	material["materialId"]->setInt(index);

	return material;
}

Material createLightMaterial(const LightParameter &mat, int index)
{
	const std::string ptx_path = ptxPath("light_hit_program.cu");
	Program ch_program = context->createProgramFromPTXFile(ptx_path, "closest_hit");

	Material material = context->createMaterial();
	material->setClosestHitProgram(0, ch_program);

	material["lightMaterialId"]->setInt(index);

	return material;
}

void updateMaterialParameters(const std::vector<MaterialParameter> &materials)
{
	MaterialParameter* dst = static_cast<MaterialParameter*>(m_bufferMaterialParameters->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
	for (size_t i = 0; i < materials.size(); ++i, ++dst) {
		MaterialParameter mat = materials[i];
		
		dst->color = mat.color;
		dst->emission = mat.emission;
		dst->metallic = mat.metallic;
		dst->subsurface = mat.subsurface;
		dst->specular = mat.specular;
		dst->specularTint = mat.specularTint;
		dst->roughness = mat.roughness;
		dst->anisotropic = mat.anisotropic;
		dst->sheen = mat.sheen;
		dst->sheenTint = mat.sheenTint;
		dst->clearcoat = mat.clearcoat;
		dst->clearcoatGloss = mat.clearcoatGloss;
		dst->brdf = mat.brdf;

		dst->dist = mat.dist;
		dst->intIOR = mat.intIOR;
		dst->extIOR = mat.extIOR;

		dst->albedoID = mat.albedoID;
	}
	m_bufferMaterialParameters->unmap();
}

void updateLightParameters(const std::vector<LightParameter> &lightParameters)
{
	LightParameter* dst = static_cast<LightParameter*>(m_bufferLightParameters->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
	for (size_t i = 0; i < lightParameters.size(); ++i, ++dst) {
		LightParameter mat = lightParameters[i];

		dst->position = mat.position;
		dst->emission = mat.emission;
		dst->radius = mat.radius;
		dst->area = mat.area;
		dst->u = mat.u;
		dst->v = mat.v;
		dst->normal = mat.normal;
		dst->lightType = mat.lightType;
	}
	m_bufferLightParameters->unmap();
}


const float randFloat(const float min, const float max)
{
	return static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) / (max - min)) + min;
}


inline int argmin(optix::float3 v)
{
	if (v.x <= v.y && v.x <= v.z) // x is the minimum
		return 0;
	if (v.y <= v.x && v.y <= v.z) // y is the minimum
		return 1;
	return 2;
}

inline float getFromIdx(optix::float3 v, int argmin)
{
	switch (argmin)
	{
	case 0:
		return v.x;
	case 1:
		return v.y;
	case 2:
		return v.z;
	default:
		return v.z;
	}
}


float swapScore(optix::float3 v, optix::float3 aabb_min, optix::float3 aabb_max, float &prob)
{
	// return distance
	optix::float3 d1 = v - aabb_min;
	optix::float3 d2 = aabb_max - v;
	int i;
	float d;

	if (getFromIdx(d1, argmin(d1)) < getFromIdx(d2, argmin(d2)))
	{ 
		i = argmin(d1);
		d = getFromIdx(d1, i);
	}
	else 
	{
		i = argmin(d2);
		d = getFromIdx(d2, i);
	}

	float half_w = 0.5f * (getFromIdx(aabb_max, i) - getFromIdx(aabb_min, i));

	float x = 2.0f * d / half_w - 1.0f;
	prob = fminf(0.5f, x / (1.0f + x * x) + 0.5f);

	return d;
}


sutil::Camera setRandomCameraParams(const optix::Aabb aabb, std::string aabb_txt_fn)
{
	FILE* aabb_txt = fopen(aabb_txt_fn.c_str(), "r");
	optix::float3 aabb_min, aabb_max; // minimum xyz, maximum xyz
	optix::float3 camera_lookat, camera_eye;
	bool indoor = true;
	const int MAXLINELEN = 256;

	if (!aabb_txt)
	{
		// aabb.txt 파일이 없다
		std::cerr << "No predefined aabb.txt file" << std::endl;
		aabb_min = aabb.m_min;
		aabb_max = aabb.m_max;
		indoor = true;
	}
	else
	{
		// aabb.txt 파일이 있다
		std::cerr << "Using predefined aabb.txt file" << std::endl;
		char line[MAXLINELEN];
		char type[MAXLINELEN] = "None";

		while (fgets(line, MAXLINELEN, aabb_txt))
		{
			sscanf(line, "type %s", type);
			sscanf(line, "xmin %f", &aabb_min.x);
			sscanf(line, "xmax %f", &aabb_max.x);
			sscanf(line, "ymin %f", &aabb_min.y);
			sscanf(line, "ymax %f", &aabb_max.y);
			sscanf(line, "zmin %f", &aabb_min.z);
			sscanf(line, "zmax %f", &aabb_max.z);
		}

		if (strcmp(type, "indoor") == 0)
			indoor = true;
		else
			indoor = false;
	}

	if (indoor)
	{
		std::cerr << "Indoor scene" << std::endl;
		camera_lookat = make_float3(
			randFloat(aabb_min.x, aabb_max.x),
			randFloat(aabb_min.y, aabb_max.y),
			randFloat(aabb_min.z, aabb_max.z)
			);
		camera_eye = make_float3(
			randFloat(aabb_min.x, aabb_max.x),
			randFloat(aabb_min.y, aabb_max.y),
			randFloat(aabb_min.z, aabb_max.z)
			);

		float prob_lookat, prob_eye, d_lookat, d_eye;
		d_lookat = swapScore(camera_lookat, aabb_min, aabb_max, prob_lookat);
		d_eye = swapScore(camera_eye, aabb_min, aabb_max, prob_eye);

		if (d_lookat > d_eye)
		{
			if (randFloat(0.0f, 1.0f) <= prob_lookat) // change lookat to eye in a low chance
			{
				optix::float3 tmp = optix::make_float3(camera_eye.x, camera_eye.y, camera_eye.z);
				camera_eye = optix::make_float3(camera_lookat.x, camera_lookat.y, camera_lookat.z);
				camera_lookat = optix::make_float3(tmp.x, tmp.y, tmp.z);
			}
		}
		else if (d_lookat == d_eye)
		{
			if (randFloat(0.0f, 1.0f) <= 0.5f) // half chance
			{
				// swap
				optix::float3 tmp = optix::make_float3(camera_eye.x, camera_eye.y, camera_eye.z);
				camera_eye = optix::make_float3(camera_lookat.x, camera_lookat.y, camera_lookat.z);
				camera_lookat = optix::make_float3(tmp.x, tmp.y, tmp.z);
			}
		}
		else
		{
			if (randFloat(0.0f, 1.0f) >= prob_eye) // change lookat to eye in a high chance
			{
				optix::float3 tmp = optix::make_float3(camera_eye.x, camera_eye.y, camera_eye.z);
				camera_eye = optix::make_float3(camera_lookat.x, camera_lookat.y, camera_lookat.z);
				camera_lookat = optix::make_float3(tmp.x, tmp.y, tmp.z);
			}
		}
	}
	else
	{
		std::cerr << "Object scene" << std::endl;
		camera_lookat = make_float3(
			randFloat(aabb_min.x, aabb_max.x),
			randFloat(aabb_min.y, aabb_max.y),
			randFloat(aabb_min.z, aabb_max.z)
			);

		optix::float3 center = 0.5f * (aabb_min + aabb_max);
		optix::float3 half_widths = 0.5f * (aabb_max - aabb_min);
		camera_eye = make_float3(
			randFloat(center.x - 5 * half_widths.x, center.x + 5 * half_widths.x),
			randFloat(center.y - half_widths.y, center.y + 5 * half_widths.y), // min.y에는 floor가 있는 경우가 많으므로 경계 확장 X
			randFloat(center.z - 5 * half_widths.z, center.z + 5 * half_widths.z)
			);
	}

	optix::float3 camera_up = optix::make_float3(
		randFloat(-0.5f, 0.5f),
		randFloat(-0.5f, 0.5f),
		randFloat(-0.5f, 0.5f));
	
	sutil::Camera camera(
		scene->properties.width, scene->properties.height, randFloat(30.0f, 60.0f),
		&camera_eye.x, &camera_lookat.x, &camera_up.x,
		context["eye"], context["U"], context["V"], context["W"]);

	if (aabb_txt)
		fclose(aabb_txt);
	return camera;
}


void setRandomMaterials()
{
	// Randomize the material parameters
	// 일단 텍스쳐 제외하고 바꾸자

	// BrdfType은 DISNEY, GLASS, ROUGHDIELECTRIC 셋 중 선택
	// DistType은 GGX로 고정
	// GLASS: intIOR [1.31 ~ 2.419], extIOR [1.0], color [0.0 ~ 1.0] x 3
	// DISNEY: color, metallic, subsurface, specular, roughness [0.0 ~ 0.6], specularTint, sheen, sheenTint, clearcoat, clearcoatGloss 나머지 모두 [0.0 ~ 1.0]
	// ROUGHDIELECTRIC: roughness [0.01 ~ 1.0] (exp), intIOR, extIOR, color

	for (size_t i = 0; i < scene->materials.size(); ++i)
	{
		scene->materials[i] = MaterialParameter();
		scene->materials[i].dist = DistType::GGX;

		float what_brdf = randFloat(0.0f, 1.0f);
		if (what_brdf < 0.9f)
		{
			scene->materials[i].brdf = BrdfType::DISNEY;
			scene->materials[i].color = optix::make_float3(randFloat(0.0f, 1.0f), randFloat(0.0f, 1.0f), randFloat(0.0f, 1.0f));
			scene->materials[i].metallic = randFloat(0.0f, 1.0f);
			scene->materials[i].subsurface = randFloat(0.0f, 1.0f);
			scene->materials[i].specular = randFloat(0.0f, 1.0f);
			scene->materials[i].roughness = randFloat(0.0f, 0.6f);
			scene->materials[i].specularTint = randFloat(0.0f, 1.0f);
			scene->materials[i].sheen = randFloat(0.0f, 1.0f);
			scene->materials[i].sheenTint = randFloat(0.0f, 1.0f);
			scene->materials[i].clearcoat = randFloat(0.0f, 1.0f);
			scene->materials[i].clearcoatGloss = randFloat(0.0f, 1.0f);
		}
		else if (what_brdf < 0.95f)
		{
			scene->materials[i].brdf = BrdfType::GLASS;
			scene->materials[i].color = optix::make_float3(randFloat(0.7f, 1.0f), randFloat(0.7f, 1.0f), randFloat(0.7f, 1.0f));
			scene->materials[i].intIOR = randFloat(1.31f, 2.419f);
			scene->materials[i].extIOR = 1.0f;
		}
		else
		{
			scene->materials[i].brdf = BrdfType::ROUGHDIELECTRIC;
			scene->materials[i].color = optix::make_float3(randFloat(0.7f, 1.0f), randFloat(0.7f, 1.0f), randFloat(0.7f, 1.0f));
			scene->materials[i].roughness = powf(10, randFloat(-2.0f, 0.0f));
			scene->materials[i].intIOR = randFloat(1.31f, 2.419f);
			scene->materials[i].extIOR = 1.0f;
		}
	}
	updateMaterialParameters(scene->materials);
	context["sysMaterialParameters"]->setBuffer(m_bufferMaterialParameters);
}


void setRandomBackground(const std::string base_hdrs, const std::vector<std::string> entries)
{
	std::string ptx_path = ptxPath("background.cu");
	context->setMissProgram(0, context->createProgramFromPTXFile(ptx_path, "miss"));
	const std::string texture_filename = base_hdrs + entries[rand() % entries.size()];
	std::cerr << texture_filename << std::endl;
	context["background_light"]->setFloat(1.0f, 1.0f, 1.0f);
	context["background_dark"]->setFloat(0.3945f, 0.0f, 0.4945f);
	context["up"]->setFloat(0.0f, 1.0f, 0.0f);
	context["option"]->setInt(1); // 1: Miss function on, 0: off (all black)

	// prevent memory leak
	context["envmap"]->getTextureSampler()->getBuffer()->destroy();
	context["envmap"]->setTextureSampler(sutil::loadTexture(context, texture_filename, optix::make_float3(1.0f)));
}


void updateAabbLights(optix::Aabb aabb)
{
	optix::float3 center = aabb.center();
	float w = aabb.extent(0) / 2.0f;
	float h = aabb.extent(1) / 2.0f;
	float d = aabb.extent(2) / 2.0f;

	for (int i : std::vector<int>({ -1, 1 }))
	{
		for (int j : std::vector<int>({ -1, 1 }))
		{
			for (int k : std::vector<int>({ -1, 1 }))
			{
				LightParameter light;
				light.position = center + optix::make_float3(i * w, j * h, k * d);
				light.emission = optix::make_float3(10000);
				light.radius = 0.1f;
				light.lightType = SPHERE;
				light.area = 4.0f * M_PIf * light.radius * light.radius;
				
				scene->lights.push_back(light);
			}
		}
	}

	std::cerr << "xmin " << center.x - w << std::endl;
	std::cerr << "xmax " << center.x + w << std::endl;
	std::cerr << "ymin " << center.y - h << std::endl;
	std::cerr << "ymax " << center.y + h << std::endl;
	std::cerr << "zmin " << center.z - d << std::endl;
	std::cerr << "zmax " << center.z + d << std::endl;

	m_bufferLightParameters = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
	m_bufferLightParameters->setElementSize(sizeof(LightParameter));
	m_bufferLightParameters->setSize(scene->lights.size());
	updateLightParameters(scene->lights);
}


optix::Aabb createGeometry(
        // output: this is a Group with two GeometryGroup children, for toggling visibility later
        optix::Group& top_group
        )
{

    const std::string ptx_path = ptxPath( "triangle_mesh.cu" );

    top_group = context->createGroup();
    top_group->setAcceleration( context->createAcceleration( "Trbvh" ) );

    int num_triangles = 0;
	size_t i,j;
    optix::Aabb aabb;
    {
        GeometryGroup geometry_group = context->createGeometryGroup();
        geometry_group->setAcceleration( context->createAcceleration( "Trbvh" ) );
        top_group->addChild( geometry_group );
		
        for (i = 0,j=0; i < scene->mesh_names.size(); ++i,++j) {
            OptiXMesh mesh;
            mesh.context = context;
            
            // override defaults
            mesh.intersection = context->createProgramFromPTXFile( ptx_path, "mesh_intersect_refine" );
            mesh.bounds = context->createProgramFromPTXFile( ptx_path, "mesh_bounds" );
            mesh.material = createMaterial(scene->materials[i], i);

            loadMesh( scene->mesh_names[i], mesh, scene->transforms[i] ); 
            geometry_group->addChild( mesh.geom_instance );

            aabb.include( mesh.bbox_min, mesh.bbox_max );

            std::cerr << scene->mesh_names[i] << ": " << mesh.num_triangles << std::endl;
            num_triangles += mesh.num_triangles;
        }
        std::cerr << "Total triangle count: " << num_triangles << std::endl;
    }
	//Lights
	{
		GeometryGroup geometry_group = context->createGeometryGroup();
		geometry_group->setAcceleration(context->createAcceleration("NoAccel"));
		top_group->addChild(geometry_group);
		
		for (i = 0; i < scene->lights.size(); ++i)
		{
			GeometryInstance instance;
			if (scene->lights[i].lightType == QUAD)
				instance = createQuad(context, createLightMaterial(scene->lights[i], i), scene->lights[i].u, scene->lights[i].v, scene->lights[i].position, scene->lights[i].normal);
			else if (scene->lights[i].lightType == SPHERE)
				instance = createSphere(context, createLightMaterial(scene->lights[i], i), scene->lights[i].position, scene->lights[i].radius);
			geometry_group->addChild(instance);
		}
		//GeometryInstance instance = createSphere(context, createMaterial(materials[j], j), optix::make_float3(150, 80, 120), 80);
		//geometry_group->addChild(instance);
	}

	

    context[ "top_object" ]->set( top_group ); 

    return aabb;
}

//------------------------------------------------------------------------------
//
//  GLFW callbacks
//
//------------------------------------------------------------------------------

struct CallbackData
{
    sutil::Camera& camera;
    unsigned int& accumulation_frame;
};

void keyCallback( GLFWwindow* window, int key, int scancode, int action, int mods )
{
    bool handled = false;

    if( action == GLFW_PRESS )
    {
        switch( key )
        {
            case GLFW_KEY_Q:
            case GLFW_KEY_ESCAPE:
                if( context )
                    context->destroy();
                if( window )
                    glfwDestroyWindow( window );
                glfwTerminate();
                exit(EXIT_SUCCESS);

            case( GLFW_KEY_S ):
            {
				const std::string outputImage = SAVE_DIR + std::string("out") + ".png";
                std::cerr << "Saving current frame to '" << outputImage << "'\n";
                sutil::writeBufferToFile( outputImage.c_str(), getOutputBuffer() );
                handled = true;
                break;
            }
            case( GLFW_KEY_F ):
            {
               CallbackData* cb = static_cast<CallbackData*>( glfwGetWindowUserPointer( window ) );
               cb->camera.reset_lookat();
               cb->accumulation_frame = 0;
               handled = true;
               break;
            }
			case ( GLFW_KEY_C ) :
			{
				CallbackData* cb = static_cast<CallbackData*>(glfwGetWindowUserPointer(window));
				optix::float3 pos = cb->camera.camera_eye();
				optix::float3 lookat = cb->camera.camera_lookat();
				optix::float3 up = cb->camera.camera_up();
				std::cerr << "Camera" << "\n";
				std::cerr << "\tposition " << pos.x << " " << pos.y << " " << pos.z << "\n";
				std::cerr << "\tlook_at " << lookat.x << " " << lookat.y << " " << lookat.z << "\n";
				std::cerr << "\tup " << up.x << " " << up.y << " " << up.z << "\n";
			}
        }
    }

    if (!handled) {
        // forward key event to imgui
        ImGui_ImplGlfw_KeyCallback( window, key, scancode, action, mods );
    }
}

void windowSizeCallback( GLFWwindow* window, int w, int h )
{
    if (w < 0 || h < 0) return;

    const unsigned width = (unsigned)w;
    const unsigned height = (unsigned)h;

    CallbackData* cb = static_cast<CallbackData*>( glfwGetWindowUserPointer( window ) );
    if ( cb->camera.resize( width, height ) ) {
        cb->accumulation_frame = 0;
    }

    sutil::resizeBuffer( getOutputBuffer(), width, height );
    sutil::resizeBuffer( context[ "accum_buffer" ]->getBuffer(), width, height );

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glViewport(0, 0, width, height);
}


//------------------------------------------------------------------------------
//
// GLFW setup and run 
//
//------------------------------------------------------------------------------

GLFWwindow* glfwInitialize( )
{
    GLFWwindow* window = sutil::initGLFW();

    // Note: this overrides imgui key callback with our own.  We'll chain this.
    glfwSetKeyCallback( window, keyCallback );

    glfwSetWindowSize( window, (int)scene->properties.width, (int)scene->properties.height);
    glfwSetWindowSizeCallback( window, windowSizeCallback );

    return window;
}


void glfwRun( GLFWwindow* window, sutil::Camera& camera, const optix::Group top_group )
{
    // Initialize GL state
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1 );
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, scene->properties.width, scene->properties.height);

    unsigned int frame_count = 0;
    unsigned int accumulation_frame = 0;
    float transmittance_log_scale = 0.0f;
	int max_depth = scene->properties.max_depth;
	lastTime = sutil::currentTime();

    // Expose user data for access in GLFW callback functions when the window is resized, etc.
    // This avoids having to make it global.
    CallbackData cb = { camera, accumulation_frame };
    glfwSetWindowUserPointer( window, &cb );

    while( !glfwWindowShouldClose( window ) )
    {

        glfwPollEvents();                                                        

        ImGui_ImplGlfw_NewFrame();

        ImGuiIO& io = ImGui::GetIO();
        
        // Let imgui process the mouse first
        if (!io.WantCaptureMouse) {

            double x, y;
            glfwGetCursorPos( window, &x, &y );

            if ( camera.process_mouse( (float)x, (float)y, ImGui::IsMouseDown(0), ImGui::IsMouseDown(1), ImGui::IsMouseDown(2) ) ) {
                accumulation_frame = 0;
            }
        }

        // imgui pushes
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,   ImVec2(0,0) );
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha,          0.6f        );
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 2.0f        );

		
        sutil::displayFps( frame_count++ );
		sutil::displaySpp( accumulation_frame );

        {
            static const ImGuiWindowFlags window_flags = 
                    ImGuiWindowFlags_NoTitleBar |
                    ImGuiWindowFlags_AlwaysAutoResize |
                    ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoScrollbar;

            ImGui::SetNextWindowPos( ImVec2( 2.0f, 70.0f ) );
            ImGui::Begin("controls", 0, window_flags );
            if ( ImGui::CollapsingHeader( "Controls", ImGuiTreeNodeFlags_DefaultOpen ) ) {
				if (ImGui::SliderInt("max depth (32)", &max_depth, 1, MAXDEPTH)) {
                    context["max_depth"]->setInt( max_depth );
                    accumulation_frame = 0;
                }
            }
            ImGui::End();
        }

		elapsedTime += sutil::currentTime() - lastTime;
		if (accumulation_frame == 0)
			elapsedTime = 0;
		sutil::displayElapsedTime(elapsedTime);
		lastTime = sutil::currentTime();

        // imgui pops
        ImGui::PopStyleVar( 3 );

        // Render main window
        context["frame"]->setUint( accumulation_frame++ );
        context->launch( 0, camera.width(), camera.height() );
        sutil::displayBufferGL( getOutputBuffer() );

        // Render gui over it
        ImGui::Render();

        glfwSwapBuffers( window );
    }
    
    destroyContext();
    glfwDestroyWindow( window );
    glfwTerminate();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
        "App Options:\n"
        "  -h | --help                  Print this usage message and exit. \n"
        "  -f | --file <output_file>    Save image to file and exit. \n"
        "  -n | --nopbo                 Disable GL interop for display buffer. (off/on) \n"
		"     | --normal                Save extra normal map of output image to file and exit. \n"
		"  -s | --scene                 Provide a scene file for rendering. \n"
		"  -p | --spp                   The number of samples per pixel. (default: 256) \n"
		"  -v | --visual                Visual mode. (off: 0, on: otherwise, default: 0) \n"
		"  -r | --random                Random camera/materials/HDR envmap. (off/on) \n"
        "App Keystrokes:\n"
        "  q  Quit\n"
        "  s  Save image to '" << SAMPLE_NAME << ".png'\n"
        "  f  Re-center camera\n"
		"  c  Show current camera attributes\n"
        "\n"
        << std::endl;

    exit(EXIT_SUCCESS);
}


int main( int argc, char** argv )
{
    bool use_pbo = true;
    std::string scene_file;
	std::string out_file;
	unsigned int num_frames = 256; // Default
	bool visual = false;
	//
	bool random = false;
	bool normal = false;
	//
    for( int i=1; i<argc; ++i )
    {
        const std::string arg(argv[i]);

        if( arg == "-h" || arg == "--help" )
        {
			printUsageAndExit(argv[0]);
        }
        else if( arg == "-f" || arg == "--file" )
        {
            if( i == argc - 1 )
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit(argv[0]);
            }
            out_file = argv[++i];
        }
		else if ( arg == "-s" || arg == "--scene" )
		{
			if (i == argc - 1)
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit(argv[0]);
			}
			scene_file = argv[++i];
		}
		else if ( arg == "-p" || arg == "--spp" )
		{
			if (i == argc - 1)
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit(argv[0]);
			}
			try
			{
				num_frames = std::stoi(argv[++i]);
				if (num_frames <= 0)
				{
					throw;
				}
			}
			catch (std::exception const &e)
			{
				std::cerr << "Option '" << arg << "' should be a positive integer value.\n";
				printUsageAndExit(argv[0]);
			}
		}
		else if (arg == "-v" || arg == "--visual")
		{
			if (i == argc - 1)
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit(argv[0]);
			}
			visual = strcmp(argv[++i], "0") == 0 ? false : true;
		}
        else if ( arg == "-n" || arg == "--nopbo" )
        {
            use_pbo = false;
        }
		else if (arg == "--normal")
		{
			normal = true;
		}
		else if (arg == "-r" || arg == "--random")
		{
			random = true;
		}
        else if( arg[0] == '-' )
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

	try
	{
		if (scene_file.empty())
		{
			std::cout << "Build succeed." << std::endl;
			// Default scene
			scene_file = sutil::samplesDir() + std::string("/data/bedroom.scene");
			scene = LoadScene(scene_file.c_str());
		}
		else
		{
			scene = LoadScene(scene_file.c_str());
		}
		SAVE_DIR = scene->dir;

		GLFWwindow* window = glfwInitialize();

		GLenum err = glewInit();

		if (err != GLEW_OK)
		{
			std::cerr << "GLEW init failed: " << glewGetErrorString(err) << std::endl;
			exit(EXIT_FAILURE);
		}

		ilInit();

		createContext(use_pbo, scene->properties.max_depth);

		// Load textures
		for (int i = 0; i < scene->texture_map.size(); i++)
		{
			Texture tex;
			Picture* picture = new Picture;
			std::string textureFilename = scene->dir + std::string(scene->texture_map[i]);
			std::cout << textureFilename << std::endl;
			if (!picture->load(textureFilename))
			{
				std::cout << "Load failed: " << textureFilename << std::endl;
			}
			tex.createSampler(context, picture);
			scene->textures.push_back(tex);
			delete picture;
		}

		// Set textures to albedo ID of materials
		for (int i = 0; i < scene->materials.size(); i++)
		{
			if (scene->materials[i].albedoID != RT_TEXTURE_ID_NULL)
			{
				scene->materials[i].albedoID = scene->textures[scene->materials[i].albedoID - 1].getId();
			}
		}

		m_bufferLightParameters = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
		m_bufferLightParameters->setElementSize(sizeof(LightParameter));
		m_bufferLightParameters->setSize(scene->lights.size());
		updateLightParameters(scene->lights);
		context["sysLightParameters"]->setBuffer(m_bufferLightParameters);

		m_bufferMaterialParameters = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
		m_bufferMaterialParameters->setElementSize(sizeof(MaterialParameter));
		m_bufferMaterialParameters->setSize(scene->materials.size());
		updateMaterialParameters(scene->materials);
		context["sysMaterialParameters"]->setBuffer(m_bufferMaterialParameters);

		context["sysNumberOfLights"]->setInt(scene->lights.size());
		optix::Group top_group;
		const optix::Aabb aabb = createGeometry(top_group);

		/* Visualize axis-aligned bounding box (aabb)
		updateAabbLights(aabb1);
		context["sysLightParameters"]->setBuffer(m_bufferLightParameters);
		context["sysNumberOfLights"]->setInt(scene->lights.size());
		const optix::Aabb aabb = createGeometry(top_group); // 이게 없어도 light은 적용됨. 눈에 안보일 뿐.
		*/

		context->validate();

		if (!scene->properties.init_eye)
			scene->properties.camera_eye = optix::make_float3(0.0f, 1.5f*aabb.extent(1), 1.5f*aabb.extent(2));
		if (!scene->properties.init_lookat)
			scene->properties.camera_lookat = aabb.center();
		if (!scene->properties.init_up)
			scene->properties.camera_up = optix::make_float3(0.0f, 1.0f, 0.0f);

		if (!random)
		{
			sutil::Camera camera(
				scene->properties.width, scene->properties.height, scene->properties.vfov,
				&scene->properties.camera_eye.x, &scene->properties.camera_lookat.x, &scene->properties.camera_up.x,
				context["eye"], context["U"], context["V"], context["W"]
				);

			if (out_file.empty() || visual)
			{
				glfwRun(window, camera, top_group);
			}
			else
			{
				// Accumulate frames for anti-aliasing
				std::cerr << "Accumulating " << num_frames << " frames ..." << std::endl;
				double startTime = sutil::currentTime();
				for (unsigned int frame = 0; frame < num_frames; ++frame)
				{
					context["frame"]->setUint(frame);
					context->launch(0, scene->properties.width, scene->properties.height);
				}
				std::cerr << "time: " << sutil::currentTime() - startTime << std::endl;
				sutil::writeBufferToFile(out_file.c_str(), getOutputBuffer());

				if (normal)
				{
					std::string normal_file(out_file);
					normal_file.insert(out_file.find('.'), "normal");
					sutil::writeBufferToFile(normal_file.c_str(), getNormalBuffer());
				}

				std::cerr << "Wrote " << out_file <<  std::endl;
				destroyContext();
			}		
		}
		else
		{
			// Axis-aligned bounding box (aabb) setup
			std::string aabb_txt_fn = scene_file.substr(0, scene_file.find_last_of("\\")) + "\\aabb.txt";
			std::cerr << "Predefined aabb file: " << aabb_txt_fn << std::endl;

			// HDR environmental map directory setup
			std::string base_hdrs("C:/Users/Dorian/data_scenes/optagen/HDRS/");
			struct dirent* entry;
			DIR* dir = opendir(base_hdrs.c_str());
			if (dir == NULL)
			{
				std::cerr << "HDRS directory not specified. \n" << std::endl;
				exit(EXIT_FAILURE);
			}
			std::vector<std::string> entries;
			while ((entry = readdir(dir)) != NULL)
			{
				if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
					entries.push_back(std::string(entry->d_name));
			}

			// Random number generator setup; should be set before run setRandomX(..) functions
			srand(static_cast <unsigned> (time(0))); 

			for (int r = 0; r < 50; r++)
			{
				sutil::Camera camera = setRandomCameraParams(aabb, aabb_txt_fn);
				setRandomMaterials();
				setRandomBackground(base_hdrs, entries);

				if (out_file.empty() || visual)
				{
					glfwRun(window, camera, top_group);
				}
				else
				{
					// Accumulate frames for anti-aliasing
					std::cerr << "Accumulating " << num_frames << " frames ..." << std::endl;
					double startTime = sutil::currentTime();
					for (unsigned int frame = 0; frame < num_frames; ++frame)
					{
						context["frame"]->setUint(frame);
						context->launch(0, scene->properties.width, scene->properties.height);
					}
					std::cerr << "time: " << sutil::currentTime() - startTime << std::endl;
					std::string new_file(out_file);
					new_file.insert(out_file.find('.'), std::to_string(r));
					sutil::writeBufferToFile(new_file.c_str(), getOutputBuffer());
					std::cerr << "Wrote " << new_file << ". fov: " << camera.vfov() << std::endl;
				}
			}
			destroyContext();
		}
        return 0;
    }
    SUTIL_CATCH( context->get() )
}

