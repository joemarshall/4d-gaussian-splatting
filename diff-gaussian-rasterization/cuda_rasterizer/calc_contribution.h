/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CALC_CONTRIBUTION_H_INCLUDED
#define CUDA_RASTERIZER_CALC_CONTRIBUTION_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace CALC_CONTRIBUTION
{
	// Perform initial steps for each Gaussian prior to rasterization.
void preprocess(int P, 
	const float* means3D,
	const float* ts,
	const glm::vec3* scales,
	const float* scales_t,
	const float scale_modifier,
	const glm::vec4* rotations,
	const glm::vec4* rotations_r,
	const float* opacities,
	int8_t *clamped,
	const float* cov3D_precomp,
	const float prefilter_var,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const float timestamp,
	const float time_duration,
	const bool rot_4d, const int gaussian_dim, 
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered);

	// Main rasterization method.
	void render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* per_pixel_error_map,
	const float* depths,
	const float4* conic_opacity,
	uint32_t* n_contrib,
	 float* out_visibility_contribution,
	float *out_weighted_contribution
	);
	
}


#endif // CUDA_RASTERIZER_CALC_CONTRIBUTION_H_INCLUDED