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
#include <fstream>
#include <dirent.h>
#include <stdint.h>

#include "cnpy.h"

#define M_REF 0
#define M_FET 1
#define M_ALL 2

using namespace optix;

const char* const SAMPLE_NAME = "OptaGen";
std::string SAVE_DIR = "";

const int NUMBER_OF_BRDF_INDICES = 4;
const int NUMBER_OF_LIGHT_INDICES = 3;
optix::Buffer m_bufferBRDFSample;
optix::Buffer m_bufferBRDFEval;
optix::Buffer m_bufferBRDFPdf;

optix::Buffer m_bufferLightSample;
optix::Buffer m_bufferMaterialParameters;
optix::Buffer m_bufferLightParameters;
Texture m_environmentTexture;

double elapsedTime = 0;
double lastTime = 0;

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------
Properties	properties;
Context		context = 0;
Scene*		scene;


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

static std::string ptxPath(const std::string& cuda_file)
{
	return
		std::string(sutil::samplesPTXDir()) +
		"/" + std::string(SAMPLE_NAME) + "_generated_" +
		cuda_file +
		".ptx";
}


static bool ends_with(const std::string& str, const std::string& suffix)
{
	return (str.size() >= suffix.size()) && (str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0);
}


static bool starts_with(const std::string& str, const std::string& prefix)
{
	return (str.size() >= prefix.size()) && (str.compare(0, prefix.size(), prefix) == 0);
}


optix::GeometryInstance createSphere(optix::Context context,
	optix::Material material,
	optix::float3 center,
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
	optix::float3 v1, optix::float3 v2, optix::float3 anchor, optix::float3 n)
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
	return context["output_buffer"]->getBuffer();
}

/*
static Buffer getNormalBuffer()
{
	return context["normal_buffer"]->getBuffer();
}
*/

static Buffer getMBFBuffer()
{
	return context["mbpf_buffer"]->getBuffer();
}


void destroyContext()
{
	if (context)
	{
		context->destroy();
		context = 0;
	}
}


void createContext(bool use_pbo, unsigned int max_depth, unsigned int num_frames)
{
	// Set up context
	context = Context::create();
	context->setRayTypeCount(2);
	context->setEntryPointCount(1);

	// Note: this sample does not need a big stack size even with high ray depths, 
	// because rays are not shot recursively.
	context->setStackSize(800);

	// Note: high max depth for reflection and refraction through glass
	context["max_depth"]->setInt(max_depth);
	context["cutoff_color"]->setFloat(0.0f, 0.0f, 0.0f);
	context["frame"]->setUint(0u);
	context["scene_epsilon"]->setFloat(1.e-3f);
	context["mbpf_frames"]->setInt(num_frames);

	Buffer buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4,
		scene->properties.width, scene->properties.height, use_pbo);
	context["output_buffer"]->set(buffer);

	//Buffer normal_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3,
	//	scene->properties.width, scene->properties.height);
	//context["normal_buffer"]->set(normal_buffer);

	// Multiple-bounce path feature buffer
	Buffer mbpf_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER,
	scene->properties.width, scene->properties.height);
	mbpf_buffer->setElementSize(sizeof(pathFeatures6) * num_frames); // a user-defined type whose size is specified with *@ref rtBufferSetElementSize.
	context["mbpf_buffer"]->set(mbpf_buffer);

	// Accumulation buffer
	Buffer accum_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
		RT_FORMAT_FLOAT4, scene->properties.width, scene->properties.height);
	context["accum_buffer"]->set(accum_buffer);

	// Ray generation program
	std::string ptx_path(ptxPath("path_trace_camera.cu"));
	Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, "pinhole_camera");
	context->setRayGenerationProgram(0, ray_gen_program);

	// Exception program
	Program exception_program = context->createProgramFromPTXFile(ptx_path, "exception");
	context->setExceptionProgram(0, exception_program);
	context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);

	// Miss program
	ptx_path = ptxPath("background.cu");
	context->setMissProgram(0, context->createProgramFromPTXFile(ptx_path, "miss"));
	const std::string texture_filename = scene->properties.envmap_fn; // scene->dir + 
	std::cerr << texture_filename << std::endl;
	context["option"]->setInt((int)(scene->properties.envmap_fn != ""));
	//context["envmap"]->setTextureSampler(sutil::loadTexture(context, texture_filename, optix::make_float3(1.0f)));

	Program prg;
	// BRDF sampling functions.
	m_bufferBRDFSample = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_BRDF_INDICES);
	int* brdfSample = (int*)m_bufferBRDFSample->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
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
	int* brdfEval = (int*)m_bufferBRDFEval->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
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
	int* brdfPdf = (int*)m_bufferBRDFPdf->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
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
	prg = context->createProgramFromPTXFile(ptxPath("light_sample.cu"), "envmap_sample");
	lightsample[LightType::ENVMAP] = prg->getId();
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
	const std::string ptx_path = ptxPath("hit_program.cu");
	Program ch_program = context->createProgramFromPTXFile(ptx_path, "closest_hit");
	Program ah_program = context->createProgramFromPTXFile(ptx_path, "any_hit");

	Material material = context->createMaterial();
	material->setClosestHitProgram(0, ch_program);
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


void updateLightParameters(std::vector<LightParameter> &lightParameters)
{
	// The environment light is expected in sysLightDefinitions[0]!
	if (scene->properties.envmap_fn != "") // HDR Environment mapping with loaded texture.
	{
		Picture* picture = new Picture;
		picture->load(scene->properties.envmap_fn);

		m_environmentTexture.createEnvironment(picture);

		delete picture;

		// Generate the CDFs for direct environment lighting and the environment texture sampler itself.
		m_environmentTexture.calculateCDF(context);

		LightParameter light;
		light.lightType = LightType::ENVMAP;
		light.area = 4.0f * M_PIf; // Unused.

		// Set the bindless texture and buffer IDs inside the LightDefinition.
		light.idEnvironmentTexture = m_environmentTexture.getId();
		light.idEnvironmentCDF_U = m_environmentTexture.getBufferCDF_U()->getId();
		light.idEnvironmentCDF_V = m_environmentTexture.getBufferCDF_V()->getId();
		light.environmentIntegral = m_environmentTexture.getIntegral(); // DAR PERF Could bake the factor 2.0f * M_PIf * M_PIf into the sysEnvironmentIntegral here.

		// Debug
		RTsize u1, u2, v;
		m_environmentTexture.getBufferCDF_U()->getSize(u1, u2);
		m_environmentTexture.getBufferCDF_V()->getSize(v);
		std::cerr << "Envmap size: " << u1 << " " << v << std::endl;

		lightParameters.insert(lightParameters.begin(), light);

		m_bufferLightParameters = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
		m_bufferLightParameters->setElementSize(sizeof(LightParameter));
		m_bufferLightParameters->setSize(lightParameters.size()); // Update the buffer size
	}

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

		dst->idEnvironmentTexture = mat.idEnvironmentTexture;
		dst->idEnvironmentCDF_U = mat.idEnvironmentCDF_U;
		dst->idEnvironmentCDF_V = mat.idEnvironmentCDF_V;
		dst->environmentIntegral = mat.environmentIntegral;
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

	optix::float3 center = 0.5f * (aabb_min + aabb_max);
	optix::float3 half_widths = 0.5f * (aabb_max - aabb_min);
	float margin = 0.7; // safe margin to prevent camera-object occlusion, etc.

	if (indoor)
	{
		std::cerr << "Indoor scene" << std::endl;

		camera_lookat = make_float3(
			randFloat(center.x - margin * half_widths.x, center.x + margin * half_widths.x),
			randFloat(center.y - margin * half_widths.y, center.y + margin * half_widths.y),
			randFloat(center.z - margin * half_widths.z, center.z + margin * half_widths.z)
			);
		camera_eye = make_float3(
			randFloat(center.x - margin * half_widths.x, center.x + margin * half_widths.x),
			randFloat(center.y - margin * half_widths.y, center.y + margin * half_widths.y),
			randFloat(center.z - margin * half_widths.z, center.z + margin * half_widths.z)
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
			randFloat(center.x - margin * half_widths.x, center.x + margin * half_widths.x),
			randFloat(center.y - margin * half_widths.y, center.y + margin * half_widths.y),
			randFloat(center.z - margin * half_widths.z, center.z + margin * half_widths.z)
			);

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
		scene->properties.width, scene->properties.height, randFloat(30.f, 60.f),
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
		int albedoID = scene->materials[i].albedoID;
		scene->materials[i] = MaterialParameter();
		scene->materials[i].albedoID = albedoID;
		scene->materials[i].dist = DistType::GGX;

		float what_brdf = randFloat(0.0f, 1.0f);
		if (what_brdf < 0.9f)
		{
			scene->materials[i].brdf = BrdfType::DISNEY;
			scene->materials[i].color = optix::make_float3(randFloat(0.0f, 1.0f), randFloat(0.0f, 1.0f), randFloat(0.0f, 1.0f));
			scene->materials[i].metallic = randFloat(0.0f, 1.0f);
			scene->materials[i].subsurface = randFloat(0.0f, 0.2f);
			scene->materials[i].specular = randFloat(0.0f, 1.0f);
			scene->materials[i].roughness = randFloat(0.0f, 0.6f);
			scene->materials[i].specularTint = randFloat(0.0f, 0.3f);
			scene->materials[i].sheen = randFloat(0.0f, 0.2f);
			scene->materials[i].sheenTint = randFloat(0.0f, 0.7f);
			scene->materials[i].clearcoat = randFloat(0.0f, 0.3f);
			scene->materials[i].clearcoatGloss = randFloat(0.0f, 1.0f);
		}
		else if (what_brdf < 0.95f)
		{
			scene->materials[i].brdf = BrdfType::GLASS;
			scene->materials[i].color = optix::make_float3(randFloat(0.7f, 1.0f), randFloat(0.7f, 1.0f), randFloat(0.7f, 1.0f));
			scene->materials[i].intIOR = randFloat(1.31f, 2.419f);
			scene->materials[i].extIOR = 1.0f;
			scene->materials[i].albedoID = RT_TEXTURE_ID_NULL;
		}
		else
		{
			scene->materials[i].brdf = BrdfType::ROUGHDIELECTRIC;
			scene->materials[i].color = optix::make_float3(randFloat(0.7f, 1.0f), randFloat(0.7f, 1.0f), randFloat(0.7f, 1.0f));
			scene->materials[i].roughness = powf(10, randFloat(-2.0f, 0.0f));
			scene->materials[i].intIOR = randFloat(1.31f, 2.419f);
			scene->materials[i].extIOR = 1.0f;
			scene->materials[i].albedoID = RT_TEXTURE_ID_NULL;
		}
	}
	updateMaterialParameters(scene->materials);
	context["sysMaterialParameters"]->setBuffer(m_bufferMaterialParameters);
}


void setRandomBackground(const std::string base_hdrs, const std::vector<std::string> entries)
{
	std::string ptx_path = ptxPath("background.cu");
	context->setMissProgram(0, context->createProgramFromPTXFile(ptx_path, "miss"));
	scene->properties.envmap_fn = base_hdrs + entries[rand() % entries.size()];
	std::cerr << scene->properties.envmap_fn << std::endl;
	context["option"]->setInt(1); // 1: Miss function on, 0: off (all black)

	if (m_environmentTexture.getWidth() != 1) { 
		// if there is a pre-assigned env light
		scene->lights.erase(scene->lights.begin());

		// prevent memory leak
		m_environmentTexture.getSampler()->getBuffer()->destroy();
		m_environmentTexture.getBufferCDF_U()->destroy();
		m_environmentTexture.getBufferCDF_V()->destroy();
	}
	context["sysLightParameters"]->getBuffer()->destroy();
	updateLightParameters(scene->lights);
	context["sysLightParameters"]->setBuffer(m_bufferLightParameters);
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

	const std::string ptx_path = ptxPath("triangle_mesh.cu");

	top_group = context->createGroup();
	top_group->setAcceleration(context->createAcceleration("Trbvh"));

	int num_triangles = 0;
	size_t i, j;
	optix::Aabb aabb;
	{
		GeometryGroup geometry_group = context->createGeometryGroup();
		geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
		top_group->addChild(geometry_group);

		for (i = 0, j = 0; i < scene->mesh_names.size(); ++i, ++j) {
			OptiXMesh mesh;
			mesh.context = context;

			// override defaults
			mesh.intersection = context->createProgramFromPTXFile(ptx_path, "mesh_intersect_refine");
			mesh.bounds = context->createProgramFromPTXFile(ptx_path, "mesh_bounds");
			mesh.material = createMaterial(scene->materials[i], i);

			loadMesh(scene->mesh_names[i], mesh, scene->transforms[i]);
			geometry_group->addChild(mesh.geom_instance);

			aabb.include(mesh.bbox_min, mesh.bbox_max);

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
			{
				instance = createQuad(context, createLightMaterial(scene->lights[i], i), scene->lights[i].u, scene->lights[i].v, scene->lights[i].position, scene->lights[i].normal);
				geometry_group->addChild(instance);
			}
			else if (scene->lights[i].lightType == SPHERE)
			{
				instance = createSphere(context, createLightMaterial(scene->lights[i], i), scene->lights[i].position, scene->lights[i].radius);
				geometry_group->addChild(instance);
			}
		}
	}



	context["top_object"]->set(top_group);

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

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	bool handled = false;

	if (action == GLFW_PRESS)
	{
		switch (key)
		{
		case GLFW_KEY_Q:
		case GLFW_KEY_ESCAPE:
			if (context)
				context->destroy();
			if (window)
				glfwDestroyWindow(window);
			glfwTerminate();
			exit(EXIT_SUCCESS);

		case(GLFW_KEY_S) :
		{
			const std::string outputImage = SAVE_DIR + std::string("out") + ".png";
			std::cerr << "Saving current frame to '" << outputImage << "'\n";
			sutil::writeBufferToFile(outputImage.c_str(), getOutputBuffer());
			handled = true;
			break;
		}
		case(GLFW_KEY_F) :
		{
			CallbackData* cb = static_cast<CallbackData*>(glfwGetWindowUserPointer(window));
			cb->camera.reset_lookat();
			cb->accumulation_frame = 0;
			handled = true;
			break;
		}
		case (GLFW_KEY_C) :
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
		ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
	}
}

void windowSizeCallback(GLFWwindow* window, int w, int h)
{
	if (w < 0 || h < 0) return;

	const unsigned width = (unsigned)w;
	const unsigned height = (unsigned)h;

	CallbackData* cb = static_cast<CallbackData*>(glfwGetWindowUserPointer(window));
	if (cb->camera.resize(width, height)) {
		cb->accumulation_frame = 0;
	}

	sutil::resizeBuffer(getOutputBuffer(), width, height);
	sutil::resizeBuffer(context["accum_buffer"]->getBuffer(), width, height);

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

GLFWwindow* glfwInitialize()
{
	GLFWwindow* window = sutil::initGLFW();

	// Note: this overrides imgui key callback with our own.  We'll chain this.
	glfwSetKeyCallback(window, keyCallback);

	glfwSetWindowSize(window, (int)scene->properties.width, (int)scene->properties.height);
	glfwSetWindowSizeCallback(window, windowSizeCallback);

	return window;
}


void glfwRun(GLFWwindow* window, sutil::Camera& camera, const optix::Group top_group)
{
	// Initialize GL state
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);
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
	glfwSetWindowUserPointer(window, &cb);

	while (!glfwWindowShouldClose(window))
	{

		glfwPollEvents();

		ImGui_ImplGlfw_NewFrame();

		ImGuiIO& io = ImGui::GetIO();

		// Let imgui process the mouse first
		if (!io.WantCaptureMouse) {

			double x, y;
			glfwGetCursorPos(window, &x, &y);

			if (camera.process_mouse((float)x, (float)y, ImGui::IsMouseDown(0), ImGui::IsMouseDown(1), ImGui::IsMouseDown(2))) {
				accumulation_frame = 0;
			}
		}

		// imgui pushes
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
		ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.6f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 2.0f);


		sutil::displayFps(frame_count++);
		sutil::displaySpp(accumulation_frame);

		{
			static const ImGuiWindowFlags window_flags =
				ImGuiWindowFlags_NoTitleBar |
				ImGuiWindowFlags_AlwaysAutoResize |
				ImGuiWindowFlags_NoMove |
				ImGuiWindowFlags_NoScrollbar;

			ImGui::SetNextWindowPos(ImVec2(2.0f, 70.0f));
			ImGui::Begin("controls", 0, window_flags);
			if (ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
				if (ImGui::SliderInt("max depth (32)", &max_depth, 1, MAXDEPTH)) {
					context["max_depth"]->setInt(max_depth);
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
		ImGui::PopStyleVar(3);

		// Render main window
		context["frame"]->setUint(accumulation_frame++);
		context->launch(0, camera.width(), camera.height());
		sutil::displayBufferGL(getOutputBuffer());

		// Render gui over it
		ImGui::Render();

		glfwSwapBuffers(window);
	}

	destroyContext();
	glfwDestroyWindow(window);
	glfwTerminate();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit()
{
	std::cerr <<
		"\n"
		"usage: OptaGen.exe [-h] [--mode MODE] --scene SCENE [--in IN] [--out OUT] [--num NUM] \n"
		"                   [--spp SPP] [--mspp MSPP] [--roc ROC] [--width WIDTH] [--visual VISUAL] \n"
		"\n"
		"OptaGen renderer... \n"
		"Copyright © 2020 by Inyoung Cho (ciy405x@kaist.ac.kr) \n"
		"All rights reserved. \n"
		"\n"
		"optional arguments:\n"
		"  -h | --help           show this help message and exit \n"
		"  -M | --mode MODE      rendering mode (0: reference image only, 1: features only, 2: reference image and features) \n"
		"  -s | --scene SCENE    scene file for rendering \n"
		"  -d | --hdr HDR        home directory for HDRIs \n"
		"  -i | --in IN          base filename for input features (.npy) \n"
		"  -o | --out OUT        base filename for output reference image (.npy) \n"
		"  -n | --num NUM        number of patches to generate \n"
		"  -p | --spp SPP        sample per pixel \n"
		"  -m | --mspp MSPP      maximum number of sample per pixel to render the reference image \n"
		"  -r | --roc ROC        if the `rate of change` of relMSE is higher than this value, stop rendering the reference image \n"
		"  -w | --width WIDTH    image width and height for training data processing \n"
		"  -v | --visual VISUAL  visual mode (0: off, 1: on) \n"
		"\n"
		"app keystrokes:\n"
		"  q  Quit\n"
		"  s  Save image to '" << SAVE_DIR + std::string("out") << ".png'\n"
		"  f  Re-center camera\n"
		"  c  Show current camera attributes\n"
		"\n"
		<< std::endl;

	exit(EXIT_SUCCESS);
}


void writeBufferToNpy(std::string filename, optix::Buffer buffer, bool ref, int num_of_frames)
{
	GLsizei width, height;
	RTsize buffer_width, buffer_height;

	float* data;
	rtBufferMap(buffer->get(), (void**)&data);
	
	buffer->getSize(buffer_width, buffer_height);
	width = static_cast<GLsizei>(buffer_width);
	height = static_cast<GLsizei>(buffer_height);

	if (ref)
	{
		assert(buffer->getElementSize() / sizeof(float) == 4);

		std::vector<float> pix(width * height * 3);
		// this buffer is upside down
		for (int j = height - 1; j >= 0; --j)
		{
			float* dst = &pix[0] + (3 * width*(height - 1 - j));
			float* src = data + 4 * width*j;
			for (int i = 0; i < width; i++)
			{
				for (int elem = 0; elem < 3; ++elem)
				{
					*dst++ = *src++;
				}

				// skip alpha (padding)
				src++;
			}

		}

		cnpy::npy_save(
			filename,
			&pix[0],
			{ buffer_width, buffer_height, 3 },
			"w");
	}
	else
	{
		assert(buffer->getElementSize() / sizeof(float) == 4 * 54);

		std::vector<float> pix(width * height * 4 * 54);
		// this buffer is upside down
		for (int j = height - 1; j >= 0; --j)
		{
			float* dst = &pix[0] + (4 * 54 * width*(height - 1 - j));
			float* src = data + 4 * 54 * width*j;
			for (int i = 0; i < width; i++)
			{
				for (int elem = 0; elem < 4 * 54; ++elem)
				{
					*dst++ = *src++;
				}

				// if spp % 4 == 0 ==> no need to pad
			}
		}

		cnpy::npy_save(
			filename,
			&pix[0],
			{ buffer_width, buffer_height, 
			(size_t)num_of_frames, 
			buffer->getElementSize() / sizeof(float) / num_of_frames },
			"w");
	}
	
	RT_CHECK_ERROR(rtBufferUnmap(buffer->get()));

	std::cerr << "[Output] " << (ref ? "(ref) " : "(feat) ") << filename << std::endl;
}


int main(int argc, char** argv)
{
	int mode = 0, num_of_patches = 1, num_of_frames = 4, max_ref_frames = 1024, width = 0;
	float rate_of_change = 0.9;
	std::string scene_file = "", hdrs_home = "", in_file = "", out_file = "";
	bool visual = false;
	bool use_pbo = false;

	std::vector<std::string> opts = {
		"-h", "--help", "-M", "--mode", "-s", "--scene",
		"-d", "--hdr", "-i", "--in", "-o", "--out", 
		"-n", "--num", "-p", "--spp", "-m", "--mspp", 
		"-r", "--roc", "-w", "--width", "-v", "--visual"
	};

	for (int i = 1; i < argc; ++i)
	{
		const std::string arg(argv[i]);

		if (arg == "-h" || arg == "--help")
		{
			printUsageAndExit();
		}
		else if (arg == "-M" || arg == "--mode")
		{
			if (i == argc - 1 || (std::find(opts.begin(), opts.end(), argv[i + 1]) != opts.end()))
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit();
			}

			try
			{
				mode = std::stoi(argv[++i]);
				if (mode != M_REF && mode != M_FET && mode != M_ALL)
				{
					throw std::exception();
				}
			}
			catch (std::exception const &e)
			{
				std::cerr << "Option '" << arg << "' should be 0, 1, or 2.\n";
				printUsageAndExit();
			}
		}
		else if (arg == "-s" || arg == "--scene")
		{
			if (i == argc - 1 || (std::find(opts.begin(), opts.end(), argv[i + 1]) != opts.end()))
			{
				std::cerr << "Option '" << arg << "' requires additional argument. \n";
				printUsageAndExit();
			}
			scene_file = argv[++i];
		}
		else if (arg == "-d" || arg == "--hdr")
		{
			if (i == argc - 1 || (std::find(opts.begin(), opts.end(), argv[i + 1]) != opts.end()))
			{
				std::cerr << "Option '" << arg << "' requires additional arguments. \n";
				printUsageAndExit();
			}
			hdrs_home = argv[++i];
		}
		else if (arg == "-i" || arg == "--in")
		{
			if (i == argc - 1 || (std::find(opts.begin(), opts.end(), argv[i + 1]) != opts.end()))
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit();
			}
			in_file = argv[++i];
			in_file = in_file.substr(0, in_file.find_last_of(".")) + ".npy";
		}
		else if (arg == "-o" || arg == "--out")
		{
			if (i == argc - 1 || (std::find(opts.begin(), opts.end(), argv[i + 1]) != opts.end()))
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit();
			}
			out_file = argv[++i];
			out_file = out_file.substr(0, out_file.find_last_of(".")) + ".npy";
		}
		else if (arg == "-n" || arg == "--num")
		{
			if (i == argc - 1 || (std::find(opts.begin(), opts.end(), argv[i + 1]) != opts.end()))
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit();
			}

			try
			{
				num_of_patches = std::stoi(argv[++i]);
				if (num_of_patches < 1)
				{
					throw std::exception();
				}
			}
			catch (std::exception const &e)
			{
				std::cerr << "Option '" << arg << "' should be a positive integer value.\n";
				printUsageAndExit();
			}
		}
		else if (arg == "-p" || arg == "--spp")
		{
			if (i == argc - 1 || (std::find(opts.begin(), opts.end(), argv[i + 1]) != opts.end()))
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit();
			}

			try
			{
				num_of_frames = std::stoi(argv[++i]);
				if (num_of_frames < 1)
				{
					throw std::exception();
				}
			}
			catch (std::exception const &e)
			{
				std::cerr << "Option '" << arg << "' should be a positive integer value.\n";
				printUsageAndExit();
			}
		}
		else if (arg == "-m" || arg == "--mspp")
		{
			if (i == argc - 1 || (std::find(opts.begin(), opts.end(), argv[i + 1]) != opts.end()))
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit();
			}

			try
			{
				max_ref_frames = std::stoi(argv[++i]);
				if (max_ref_frames < 1)
				{
					throw std::exception();
				}
			}
			catch (std::exception const &e)
			{
				std::cerr << "Option '" << arg << "' should be a positive integer value.\n";
				printUsageAndExit();
			}
		}
		else if (arg == "-r" || arg == "--roc")
		{
			if (i == argc - 1 || (std::find(opts.begin(), opts.end(), argv[i + 1]) != opts.end()))
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit();
			}

			try
			{
				rate_of_change = std::stof(argv[++i]);
				if (rate_of_change <= 0.0 || rate_of_change >= 1.0)
				{
					throw std::exception();
				}
			}
			catch (std::exception const &e)
			{
				std::cerr << "Option '" << arg << "' should be bound in (0.0, 1.0).\n";
				printUsageAndExit();
			}
		}
		else if (arg == "-w" || arg == "--width")
		{
			if (i == argc - 1 || (std::find(opts.begin(), opts.end(), argv[i + 1]) != opts.end()))
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit();
			}

			try
			{
				width = std::stoi(argv[++i]);
				if (width <= 0)
				{
					throw std::exception();
				}
			}
			catch (std::exception const &e)
			{
				std::cerr << "Option '" << arg << "' should be a positive interger value.\n";
				printUsageAndExit();
			}
		}
		else if (arg == "-v" || arg == "--visual")
		{
			if (i == argc - 1 || (std::find(opts.begin(), opts.end(), argv[i + 1]) != opts.end()))
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit();
			}
			visual = strcmp(argv[++i], "0") == 0 ? false : true;
		}
		else if (arg[0] == '-')
		{
			std::cerr << "Unknown option '" << arg << "'. \n";
			printUsageAndExit();
		}
		else
		{
			std::cerr << "Unknown command. \n";
			printUsageAndExit();
		}
	}
	
	if (max_ref_frames < num_of_frames)
	{
		std::cerr << "Option '--mspp' should be larger than '--spp'. \n";
		std::cerr << "(MSPP " << max_ref_frames << ", SPP " << num_of_frames << ")";
		printUsageAndExit();
	}

	if (scene_file.empty())
	{
		std::cerr << "Option '--scene' is required. \n";
		printUsageAndExit();
	}

	if (mode == M_FET && in_file.empty())
	{
		std::cerr << "Option '--in' is required in the feature-only mode. \n";
		printUsageAndExit();
	}

	if (mode == M_ALL && (in_file.empty() || out_file.empty()))
	{
		std::cerr << "Option '--in' and '--out' are required in the image-and-feature mode. \n";
		printUsageAndExit();
	}

	try
	{
		scene = LoadScene(scene_file.c_str());
		if (width != 0)
		{
			scene->properties.width = width;
			scene->properties.height = width;
		}
		SAVE_DIR = scene->dir;


		GLFWwindow* window;
		GLenum err;
		if (visual || (in_file.empty() && out_file.empty()))
		{
			window = glfwInitialize();
			err = glewInit();

			if (err != GLEW_OK)
			{
				std::cerr << "GLEW init failed: " << glewGetErrorString(err) << std::endl;
				exit(EXIT_FAILURE);
			}
		}

		ilInit();

		createContext(use_pbo, scene->properties.max_depth, num_of_frames);

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
		
		if (visual || (in_file.empty() && out_file.empty()))
		{
			std::cerr << "[Mode] visual";

			sutil::Camera camera(
				scene->properties.width, scene->properties.height, scene->properties.vfov,
				&scene->properties.camera_eye.x, &scene->properties.camera_lookat.x, &scene->properties.camera_up.x,
				context["eye"], context["U"], context["V"], context["W"]
				);

			glfwRun(window, camera, top_group);
		}
		else
		{
			if (num_of_patches == 1)
			{
				if (mode == M_REF)
					std::cerr << "[Mode] reference-only" << "\n";
				else if (mode == M_FET)
					std::cerr << "[Mode] feature-only" << "\n";
				else
					std::cerr << "[Mode] reference-and-feature" << "\n";

				sutil::Camera camera(
					scene->properties.width, scene->properties.height, scene->properties.vfov,
					&scene->properties.camera_eye.x, &scene->properties.camera_lookat.x, &scene->properties.camera_up.x,
					context["eye"], context["U"], context["V"], context["W"]
					);

				if (mode == M_REF)
					std::cerr << "[Samples] " << max_ref_frames << " (per-pixel)\n";
				else if (mode == M_FET)
					std::cerr << "[Samples] " << num_of_frames << " (per-pixel)\n";
				else
					std::cerr << "[Samples] (feat) " << num_of_frames << ", (ref) " << max_ref_frames << " (per-pixel)\n";

				std::cerr << "[Film size] " << scene->properties.width << ", " << scene->properties.height << "\n";

				double startTime = sutil::currentTime();

				if (mode == M_FET || mode == M_ALL)
				{
					for (unsigned int frame = 0; frame < num_of_frames; ++frame)
					{
						context["frame"]->setUint(frame);
						context->launch(0, scene->properties.width, scene->properties.height);
					}
					std::cerr << "[Elapsed time] (feat) " << sutil::currentTime() - startTime << "s\n";
					
					/*
					RTbuffer buf = getMBFBuffer()->get();
					float* data;
					rtBufferMap(buf, (void**)&data);
					RTsize width, height;
					getMBFBuffer()->getSize(width, height);
					cnpy::npy_save(in_file,
						(float *)data,
						{ width, height, (size_t)num_of_frames, getMBFBuffer()->getElementSize() / sizeof(float) / num_of_frames },
						"w");
					rtBufferUnmap(buf);

					std::cerr << "[Output] (feat) " << in_file << "\n";
					*/

					writeBufferToNpy(in_file, getMBFBuffer(), false, num_of_frames);

					startTime = sutil::currentTime();
				}

				if (mode == M_REF || mode == M_ALL)
				{
					for (unsigned int frame = 0; frame < max_ref_frames; ++frame)
					{
						context["frame"]->setUint(frame);
						context->launch(0, scene->properties.width, scene->properties.height);
					}
					std::cerr << "[Elapsed time] (ref) " << sutil::currentTime() - startTime << "s\n";

					/*
					RTbuffer buf = getOutputBuffer()->get();
					float* data;
					rtBufferMap(buf, (void**)&data);
					RTsize width, height;
					getOutputBuffer()->getSize(width, height);
					cnpy::npy_save(out_file,
						(float *)data,
						{ width, height, getOutputBuffer()->getElementSize() / sizeof(float) },
						"w");
					rtBufferUnmap(buf);

					std::cerr << "[Output] (ref) " << out_file << std::endl;
					*/
					
					writeBufferToNpy(out_file, getOutputBuffer(), true, num_of_frames);
				}

				destroyContext();
			}
			else
			{
				if (mode == M_REF)
					std::cerr << "[Mode] reference-only" << "\n";
				else if (mode == M_FET)
					std::cerr << "[Mode] feature-only" << "\n";
				else
					std::cerr << "[Mode] reference-and-feature" << "\n";

				std::string in_fn;
				std::string out_fn;
				std::string aabb_txt_fn = scene_file.substr(0, scene_file.find_last_of("\\")) + "\\aabb.txt";
				srand(static_cast <unsigned> (time(0)));

				struct dirent* entry;
				DIR* dir = opendir(hdrs_home.c_str());
				if (dir == NULL)
				{
					std::cerr << "HDRS directory not specified! " << std::endl;
					exit(EXIT_FAILURE);
				}
				std::vector<std::string> entries;
				while ((entry = readdir(dir)) != NULL)
				{
					if (ends_with(entry->d_name, ".hdr"))
						entries.push_back(std::string(entry->d_name));
				}

				for (int r = 0; r < num_of_patches; r++)
				{
					sutil::Camera camera = setRandomCameraParams(aabb, aabb_txt_fn);
					setRandomMaterials();
					//if (r % 100 == 0)
					//	setRandomBackground(hdrs_home, entries);

					if (mode == M_REF)
						std::cerr << "[Frames] " << max_ref_frames << "\n";
					else if (mode == M_FET)
						std::cerr << "[Frames] " << num_of_frames << "\n";
					else
						std::cerr << "[Frames] (feat) " << num_of_frames << ", (ref) " << max_ref_frames << "\n";

					std::cerr << "[Film size] " << scene->properties.width << ", " << scene->properties.height << "\n";

					double startTime = sutil::currentTime();
					if (mode == M_FET || mode == M_ALL)
					{
						for (unsigned int frame = 0; frame < num_of_frames; ++frame)
						{
							context["frame"]->setUint(frame);
							context->launch(0, scene->properties.width, scene->properties.height);
						}
						std::cerr << "[Elapsed time] (feat) " << sutil::currentTime() - startTime << "\n";

						in_fn = in_file.substr(0, in_file.find('.')) + "_" + std::to_string(r) + ".npy";

						/*
						RTbuffer buf = getMBFBuffer()->get();
						float* data;
						rtBufferMap(buf, (void**)&data);
						RTsize width, height;
						getMBFBuffer()->getSize(width, height);
						cnpy::npy_save(in_fn,
							(float *)data,
							{ width, height, (size_t)num_of_frames, getMBFBuffer()->getElementSize() / sizeof(float) / num_of_frames },
							"w");
						rtBufferUnmap(buf);

						std::cerr << "[Output] (feat) " << in_fn << "\n";
						*/

						writeBufferToNpy(in_fn, getMBFBuffer(), false, num_of_frames);

						startTime = sutil::currentTime();
					}

					if (mode == M_REF || mode == M_ALL)
					{
						for (unsigned int frame = 0; frame < max_ref_frames; ++frame)
						{
							context["frame"]->setUint(frame);
							context->launch(0, scene->properties.width, scene->properties.height);
						}
						std::cerr << "[Elapsed time] (ref) " << sutil::currentTime() - startTime << "\n";

						out_fn = out_file.substr(0, out_file.find('.')) + "_" + std::to_string(r) + ".npy";
						
						/*
						RTbuffer buf = getOutputBuffer()->get();
						float* data;
						rtBufferMap(buf, (void**)&data);
						RTsize width, height;
						getOutputBuffer()->getSize(width, height);
						cnpy::npy_save(out_fn,
							(float *)data,
							{ width, height, getOutputBuffer()->getElementSize() / sizeof(float) },
							"w");
						rtBufferUnmap(buf);

						std::cerr << "[Output] (ref) " << out_fn << std::endl;
						*/

						writeBufferToNpy(out_fn, getOutputBuffer(), true, num_of_frames);
					}
				}

				destroyContext();
			}
		}

		return 0;
	}
	SUTIL_CATCH(context->get())
}

