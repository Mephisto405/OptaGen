#pragma once

#ifndef PROPERTIES_H
#define PROPERTIES_H

#define MAXDEPTH 32
#define MINDEPTH 1
#define MINW 16
#define MAXW 1920
#define MINH 16
#define MAXH 1080
#define MINFOV 1

struct Properties
{
	Properties() : width(1280), height(720), vfov(35.0f), max_depth(3), 
		init_eye(false), init_lookat(false), init_up(false), envmap_fn("") {}
	int width;
	int height;
	float vfov;
	unsigned int max_depth;
	bool init_eye;					/* is camera_eye initialized by the .scene file */
	bool init_lookat;				/* is camera_lookat initialized by the .scene file */
	bool init_up;					/* is camera_up initialized by the .scene file */
	optix::float3 camera_eye;
	optix::float3 camera_lookat;
	optix::float3 camera_up;
	std::string envmap_fn;			/* filename of the environmental map */
};

struct CameraParams
{
	optix::float3 camera_eye;
	optix::float3 camera_lookat;
	optix::float3 camera_up;
};

#endif
