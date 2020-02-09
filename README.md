# OptaGen: OptiX-based Autonomous Data Generation Tool (work in progress)

## This work has started on top of a great Optix-based renderer https://github.com/knightcrawler25/Optix-PathTracer. 

## Disclaimer: This work is in progress. Therefore, the code just works fine but not well-structured (so the readme file is). After implementing all the necessary functions, I plan to refine the code structure. Thanks.

## Features (star signs indicate knightcrawler25's original implementation)
- Rendering

-- Microfacet models (GGX/Phong/Beckmann) for reflection and refraction

-- *Disney BRDF

-- *Simple glass

- Automated data generation

-- Randomization of materials, camera parameters, HDR environmental maps, and lighting

- Last-and-least

-- xml2scene parser for Mitsuba-oriented scenes

-- drag-drop input file

-- (not physically-based) tinted glass

-- HDR environmental map support

-- camera config. preset

## TO-DO
- Algorithm support

-- Multichannel rendering: optixPathTracer.cpp에 183번째 줄에 output_buffer라는 개념 등장

-- reconstruction filtering (tent filter): writeBufferToFile 에서 구현 ([1/2 1 1/2][1/2 1 1/2]'를 normalize해서 쓰면 됨)

- Material support (렌더링 퀄리티에 영향 크게 주는 중요도 순서)

-- Multiple importance sampling (MIS) for HDR environmental maps

-- opacity (mask): 반투명 물체 구현이 시급

-- conductor => eta, k 값 적용 가능하도록 (그래야 알루미늄, 철 등을 구분 가능) (혹시 eta, k 필요 없이도 sheen, clearcoat 로 구현 가능?) (color를 강제로 넣는 것도 괜찮을 듯)

-- thindielectric => car, car2 scene에서 crucial

-- coating: (for car, car2 scenes) IOR, thickness

## Requirements

parse >= 1.12.1