/*Copyright (c) 2016 Miles Macklin

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgement in the product documentation would be
appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.*/

#include "sceneLoader.h"
#include <filesystem>

static const int kMaxLineLength = 2048;

std::string getDir(const char* filename)
{
	std::string fn(filename);
	size_t pos_slash = fn.find_last_of("\\");
	return (pos_slash != std::string::npos) ? fn.substr(0, pos_slash + 1) : std::string(sutil::samplesDir()) + "/data/";
}

float clip(float &x, float min, float max)
{
	x = (x < min) ? min : ((x > max ? max : x));
	return x;
}

Scene* LoadScene(const char* filename)
{
	Scene *scene = new Scene;
	int tex_id = 0;
	FILE* file = fopen(filename, "r");
	scene->dir = getDir(filename);

	if (!file)
	{
		printf("Couldn't open %s for reading.", filename);
		return NULL;
	}

	std::map<std::string, MaterialParameter> materials_map;
	std::map<std::string, int> texture_ids;

	char line[kMaxLineLength];

	while (fgets(line, kMaxLineLength, file))
	{
		// skip comments
		if (line[0] == '#')
			continue;

		// name used for materials and meshes
		char name[kMaxLineLength] = { 0 };


		//--------------------------------------------
		// Material

		if (sscanf(line, " material %s", name) == 1)
		{
			printf("%s", line);

			MaterialParameter material;
			char tex_name[kMaxLineLength] = "None";
			char brdf_type[20] = "None";
			char dist_type[20] = "None";

			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;

				sscanf(line, " name %s", name);
				sscanf(line, " color %f %f %f", &material.color.x, &material.color.y, &material.color.z);
				sscanf(line, " albedoTex %s", &tex_name);
				sscanf(line, " emission %f %f %f", &material.emission.x, &material.emission.y, &material.emission.z);

				sscanf(line, " metallic %f", &material.metallic);
				sscanf(line, " subsurface %f", &material.subsurface);
				sscanf(line, " specular %f", &material.specular);
				sscanf(line, " specularTint %f", &material.specularTint);
				sscanf(line, " roughness %f", &material.roughness);
				sscanf(line, " sheen %f", &material.sheen);
				sscanf(line, " sheenTint %f", &material.sheenTint);
				sscanf(line, " clearcoat %f", &material.clearcoat);
				sscanf(line, " clearcoatGloss %f", &material.clearcoatGloss);
				sscanf(line, " brdf %s", brdf_type);

				// params for rough dielectrics
				sscanf(line, " intIOR %f", &material.intIOR);
				sscanf(line, " extIOR %f", &material.extIOR);
				sscanf(line, " dist %s", dist_type);

				if (strcmp(brdf_type, "DISNEY") == 0 || strcmp(brdf_type, "0") == 0)
					material.brdf = DISNEY;
				else if (strcmp(brdf_type, "GLASS") == 0 || strcmp(brdf_type, "1") == 0)
					material.brdf = GLASS;
				else if (strcmp(brdf_type, "LAMBERT") == 0 || strcmp(brdf_type, "2") == 0)
					material.brdf = LAMBERT;
				else if (strcmp(brdf_type, "ROUGHDIELECTRIC") == 0 || strcmp(brdf_type, "3") == 0)
					material.brdf = ROUGHDIELECTRIC;
				else
					material.brdf = DISNEY;

				if (strcmp(dist_type, "Beckmann") == 0 ||
					strcmp(dist_type, "beckmann") == 0 ||
					strcmp(dist_type, "0") == 0)
					material.dist = Beckmann;
				else if (strcmp(dist_type, "GGX") == 0 ||
					strcmp(dist_type, "ggx") == 0 ||
					strcmp(dist_type, "1") == 0)
					material.dist = GGX;
				else if (strcmp(dist_type, "Phong") == 0 ||
					strcmp(dist_type, "phong") == 0 ||
					strcmp(dist_type, "2") == 0)
					material.dist = Phong;
				else
					material.dist = GGX;

				// clipping
				clip(material.color.x, 0.0f, 1.0f);
				clip(material.metallic, 0.0f, 1.0f);
				clip(material.subsurface, 0.0f, 1.0f);
				clip(material.specular, 0.0f, 1.0f);
				clip(material.specularTint, 0.0f, 1.0f);
				clip(material.sheen, 0.0f, 1.0f);
				clip(material.sheenTint, 0.0f, 1.0f);
				clip(material.clearcoat, 0.0f, 1.0f);
				clip(material.clearcoatGloss, 0.0f, 1.0f);
				clip(material.roughness, 0.0f, 1.0f);

				if (material.brdf == DISNEY)
				{
					if (material.roughness < 0.004f)
					{
						printf("Cannot create a GGX distribution with roughness<0.004 (clamped to 0.004)."
							"Please use the corresponding smooth reflectance model to get zero roughness. \n");
						material.roughness = 0.004f;
					}
				}
				else if (material.brdf == ROUGHDIELECTRIC)
				{
					if (material.roughness < 0.023f)
					{
						printf("Cannot create a GGX distribution with roughness<0.023 (clamped to 0.023)."
							"Please use the corresponding smooth reflectance model to get zero roughness. \n");
						material.roughness = 0.023f;
					}
				}

			}

			// Check if texture is already loaded
			if (texture_ids.find(tex_name) != texture_ids.end()) // Found Texture
			{
				material.albedoID = texture_ids[tex_name];
			}
			else if (strcmp(tex_name, "None") != 0)
			{
				tex_id++;
				texture_ids[tex_name] = tex_id;
				scene->texture_map[tex_id - 1] = tex_name;
				material.albedoID = tex_id;
			}

			// add material to map
			materials_map[name] = material;
		}

		//--------------------------------------------
		// Light

		if (strstr(line, "light"))
		{
			LightParameter light;
			optix::float3 v1, v2;
			char light_type[20] = "None";

			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;

				sscanf(line, " position %f %f %f", &light.position.x, &light.position.y, &light.position.z);
				sscanf(line, " emission %f %f %f", &light.emission.x, &light.emission.y, &light.emission.z);
				sscanf(line, " normal %f %f %f", &light.normal.x, &light.normal.y, &light.normal.z);

				sscanf(line, " radius %f", &light.radius);
				sscanf(line, " v1 %f %f %f", &v1.x, &v1.y, &v1.z);
				sscanf(line, " v2 %f %f %f", &v2.x, &v2.y, &v2.z);
				sscanf(line, " type %s", light_type);
			}

			if (strcmp(light_type, "Quad") == 0 || strcmp(light_type, "1") == 0)
			{
				light.lightType = QUAD;
				light.u = v1 - light.position;
				light.v = v2 - light.position;
				light.area = optix::length(optix::cross(light.u, light.v));
				light.normal = optix::normalize(optix::cross(light.u, light.v));
			}
			else if (strcmp(light_type, "Sphere") == 0 || strcmp(light_type, "0") == 0)
			{
				light.lightType = SPHERE;
				// light.normal = optix::normalize(light.normal);
				light.area = 4.0f * M_PIf * light.radius * light.radius;
			}

			scene->lights.push_back(light);
		}

		//--------------------------------------------
		// Properties

		//Defaults
		Properties prop;

		if (strstr(line, "properties"))
		{
			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;

				char envmap_fn[2048];

				if (sscanf(line, " width %i", &prop.width) != 0)
					prop.width = (prop.width < MINW) ? MINW : ((prop.width > MAXW ? MAXW : prop.width));
				if (sscanf(line, " height %i", &prop.height) != 0)
					prop.height = (prop.height < MINH) ? MINH : ((prop.height > MAXH ? MAXH : prop.height));
				if (sscanf(line, " fov %f", &prop.vfov) != 0)
					prop.vfov = (prop.vfov < MINFOV) ? MINFOV : prop.vfov;
				if (sscanf(line, " max_depth %u", &prop.max_depth) != 0)
					prop.max_depth = (prop.max_depth < MINDEPTH) ? MINDEPTH : ((prop.max_depth > MAXDEPTH ? MAXDEPTH : prop.max_depth));

				if (sscanf(line, " position %f %f %f", &prop.camera_eye.x, &prop.camera_eye.y, &prop.camera_eye.z) != 0)
					prop.init_eye = true;
				if (sscanf(line, " look_at %f %f %f", &prop.camera_lookat.x, &prop.camera_lookat.y, &prop.camera_lookat.z) != 0)
					prop.init_lookat = true;
				if (sscanf(line, " up %f %f %f", &prop.camera_up.x, &prop.camera_up.y, &prop.camera_up.z) != 0)
					prop.init_up = true;
				if (sscanf(line, " envmap %s", envmap_fn) == 1) {
					prop.envmap_fn = std::string(envmap_fn);
				}

			}

			scene->properties = prop;
		}

		//--------------------------------------------
		// Mesh

		if (strstr(line, "mesh"))
		{
			optix::Matrix4x4 xform = optix::Matrix4x4::identity();

			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;

				int count = 0;

				char path[2048];
				float data[4 * 4];

				/*
				if (sscanf(line, " file %s", path) == 1)
				{
				const optix::Matrix4x4 xform = optix::Matrix4x4::identity();// optix::Matrix4x4::rotate(-M_PIf / 2.0f, optix::make_float3(0.0f, 1.0f, 0.0f)
				scene->mesh_names.push_back(std::string(sutil::samplesDir()) + "/data/" + path);
				scene->transforms.push_back(xform);
				}
				*/


				if (sscanf(line, " file %s", path) == 1)
				{
					scene->mesh_names.push_back(scene->dir + path);
				}

				if (sscanf(line, " transform %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f",
					&data[0], &data[1], &data[2], &data[3],
					&data[4], &data[5], &data[6], &data[7],
					&data[8], &data[9], &data[10], &data[11],
					&data[12], &data[13], &data[14], &data[15]) != 0)
				{
					xform = optix::Matrix4x4(data);
				}

				if (sscanf(line, " material %s", path) == 1)
				{
					// look up material in dictionary
					if (materials_map.find(path) != materials_map.end())
					{
						MaterialParameter copy_mat(materials_map[path]);
						scene->materials.push_back(copy_mat);
					}
					else
					{
						printf("Could not find material %s\n", path);
					}
				}
			}

			scene->transforms.push_back(xform);
		}
	}
	return scene;
}