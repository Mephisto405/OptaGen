# OptaGen: OptiX-based Autonomous Data Generation Tool

OptaGen is a tool that helps you organize training datasets for deep-learning based Monte Carlo image denoisers.

This work has started on top of a great [Optix-based renderer](https://github.com/knightcrawler25/Optix-PathTracer).

## Disclaimer

The code and document is not polished yet. I will continue to update the code to make it more clean, maintainalbe and robust.

## Rendering Resources for OptiX

[This Google Drive link](https://drive.google.com/open?id=1FKiPY7VtGvwdirNgEH6yYpHk_kPlnybE) offers 25 different 3D scenes and 30 HDRIs. Most of these scenes were made by artists on [Blend Swap](https://blendswap.com/). Then they were cleaned up by [Benedikt Bitterli](https://benedikt-bitterli.me/resources/). On top of that, I massaged some geometries, textures, and OBJ so that the scenes are compatible with the [Optix-based renderer](https://github.com/knightcrawler25/Optix-PathTracer).

## How-to-Build

Please take a look at [how-to-build.txt](./how-to-build.txt)
*I will write more detailed step-by-step description soon.

## How-to-Run

Please take a look at the main [data generator code](./scripts/optagen.py) and the [feature visualizer code](./scripts/vis_feat.py)
*I will write more detailed comment soon.

### Data Configurations

25 scenes = 15 outdoor + 10 indoor = 18 training + 7 test
30 HDRIs = 10 city theme outdoor + 10 nature theme outdoor + 10 indoor

## Features 

star signs(*) indicate the original implementation of the author of [Optix-based renderer](https://github.com/knightcrawler25/Optix-PathTracer).

### Part 1: Renderer

- [x] Microfacet models (GGX/Phong/Beckmann) for reflection and refraction

- [x] Multiple importance sampling for HDR environmental map

- [x] *Disney BRDF

- [x] *Simple glass

- [ ] Reconstruction filtering (tent filter): writeBufferToFile 에서 구현 ([1/2 1 1/2][1/2 1 1/2]'를 normalize해서 쓰면 됨)

- [ ] opacity (mask): 반투명 물체 구현이 시급

- [ ] conductor => eta, k 값 적용 가능하도록 (그래야 알루미늄, 철 등을 구분 가능) (혹시 eta, k 필요 없이도 sheen, clearcoat 로 구현 가능?) (color를 강제로 넣는 것도 괜찮을 듯)

- [ ] thindielectric => car, car2 scene에서 crucial

- [ ] coating: (for car, car2 scenes) IOR, thickness

### Part 2: Data Generator

- [x] Multichannel rendering (.npy)

- [x] Randomization of materials, camera parameters, HDR environmental maps, and lighting

### Part 3: Auxiliary Features

- [x] xml2scene parser for Mitsuba-oriented scenes

- [x] drag-drop input file

- [x] (not physically-based) tinted glass

- [x] camera config. preset