# OptaGen: OptiX-based Autonomous Data Generation Tool (work in progress)

## This work has started on top of a great [Optix-based renderer] (https://github.com/knightcrawler25/Optix-PathTracer).

## Disclaimer
This work is in progress. Therefore, the code just works fine but not well-structured (so the readme file is). After implementing all the necessary functions, I plan to refine the code structure. Thanks.

## Features (star signs indicate knightcrawler25's original implementation)
1. Rendering

- [x] Microfacet models (GGX/Phong/Beckmann) for reflection and refraction

- [x] Multiple importance sampling for HDR environmental map

- [x] *Disney BRDF

- [x] *Simple glass

- [ ] Reconstruction filtering (tent filter): writeBufferToFile 에서 구현 ([1/2 1 1/2][1/2 1 1/2]'를 normalize해서 쓰면 됨)

- [ ] opacity (mask): 반투명 물체 구현이 시급

- [ ] conductor => eta, k 값 적용 가능하도록 (그래야 알루미늄, 철 등을 구분 가능) (혹시 eta, k 필요 없이도 sheen, clearcoat 로 구현 가능?) (color를 강제로 넣는 것도 괜찮을 듯)

- [ ] thindielectric => car, car2 scene에서 crucial

- [ ] coating: (for car, car2 scenes) IOR, thickness

2. Automated data generation

- [x] Multichannel rendering

- [x] Randomization of materials, camera parameters, HDR environmental maps, and lighting

3. Last-and-least

- [x] xml2scene parser for Mitsuba-oriented scenes

- [x] drag-drop input file

- [x] (not physically-based) tinted glass

- [x] camera config. preset

## Requirements
parse >= 1.12.1