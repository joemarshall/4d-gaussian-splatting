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
#include <iostream>
#include "calc_contribution.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;



// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P,
	const float* orig_points,
	const float* ts,
	const glm::vec3* scales,
	const float* scales_t,
	const float scale_modifier,
	const glm::vec4* rotations,
	const glm::vec4* rotations_r,
	const float* opacities,
	bool* clamped,
	const float* cov3D_precomp,
	const float prefilter_var,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const float timestamp,
	const float time_duration,
	const bool rot_4d, const int gaussian_dim, 
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	tiles_touched[idx] = 0;

	// // Perform near culling, quit if outside.
	// float3 p_view;
	// if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
	// 	return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float opacity = opacities[idx];

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters.
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else if (rot_4d) // gaussian_dim == 4 && rot_4d
	{
		bool time_mask=true;
		computeCov3D_conditional(scales[idx], scales_t[idx], scale_modifier,
			rotations[idx], rotations_r[idx], cov3Ds + idx * 6, p_orig, ts[idx], timestamp, idx, time_mask, opacity,
			prefilter_var);
		if (!time_mask) return;
		cov3D = cov3Ds + idx * 6;
		// out_means3D[idx*3+0]=p_orig.x;
		// out_means3D[idx*3+1]=p_orig.y;
		// out_means3D[idx*3+2]=p_orig.z;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
		if (gaussian_dim == 4){  // no rot_4d
            float dt = ts[idx]-timestamp;
            float sigma = scales_t[idx] * scale_modifier;
		    float marginal_t = __expf(-0.5*dt*dt/((prefilter_var > 0.0) ? (prefilter_var + sigma) : sigma));
		    if (marginal_t <= 0.05) return;
		    opacity *= marginal_t;
		}
	}

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(p_orig, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	float4 con_o = { conic.x, conic.y, conic.z, opacity };

	const float mult = 0.5f;

	#if FASTGS_CULLING
		uint32_t tiles_count = duplicateToTilesTouched(point_image, con_o, grid, mult, 0, 0, 0, nullptr, nullptr);
		if (tiles_count == 0)
			return;
	#else

		uint2 rect_min, rect_max;
		getRect(point_image, my_radius, rect_min, rect_max, grid);
		if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0 || ((int)my_radius <= 0.4))
			return;
		uint32_t tiles_count = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	
	#endif

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity };
	tiles_touched[idx] = tiles_count;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ per_pixel_error_map,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
//	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ out_visibility_contribution,
	float* __restrict__ out_weighted_contribution
)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float Flow[2] = { 0 };
	float D = { 0 };

	float per_pixel_error = 1.0f;
	if(per_pixel_error_map!=nullptr)
	{
		per_pixel_error = per_pixel_error_map[pix_id];
	}


	// Iterate over batches until all done or range is complete
	// or alpha is < 0.01 (i.e. no more gaussians will contribute significantly to this pixel).
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;

			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			D += depths[collected_id[j]] * alpha * T;

			T = test_T;
			if( out_visibility_contribution!=nullptr){
				atomicAdd(&(out_visibility_contribution[collected_id[j]]), T);
				if(out_weighted_contribution!=nullptr && per_pixel_error_map!= nullptr){
					float weighted_contribution = T * per_pixel_error;
					atomicAdd(&(out_weighted_contribution[collected_id[j]]), weighted_contribution);
				}

			}

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// no rendering, we're just getting contributions (above)
	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	// if (inside)
	// {
	// 	final_T[pix_id] = T;
	// 	n_contrib[pix_id] = last_contributor;
	// 	for (int ch = 0; ch < CHANNELS; ch++)
	// 		out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	// 	for (int ch = 0; ch < 2; ch++)
	// 		out_flow[ch * H * W + pix_id] = Flow[ch];
	// 	out_depth[pix_id] = D;
	// }
}

void CALC_CONTRIBUTION::render(
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
)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		per_pixel_error_map,
		depths,
		conic_opacity,
		n_contrib,
		out_visibility_contribution,
		out_weighted_contribution);
}

void CALC_CONTRIBUTION::preprocess(int P, 
	const float* means3D,
	const float* ts,
	const glm::vec3* scales,
	const float* scales_t,
	const float scale_modifier,
	const glm::vec4* rotations,
	const glm::vec4* rotations_r,
	const float* opacities,
	bool* clamped,
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
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, 
		means3D,
		ts,
		scales,
		scales_t,
		scale_modifier,
		rotations,
		rotations_r,
		opacities,
		clamped,
		cov3D_precomp,
		prefilter_var,
		viewmatrix,
		projmatrix,
		cam_pos,
		timestamp,
		time_duration,
		rot_4d, gaussian_dim, 
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		means2D,
		depths,
		cov3Ds,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}