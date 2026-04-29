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

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)
#define MY_PI 3.14159265

// if this = 1 then we only render to buckets if the tile 
// is covered by a lot of gaussians
// otherwise we render to all tiles within radius of the gaussian
#define FASTGS_CULLING 1


// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(
	float3 p_orig,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f) // || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

// This is built upon Speedy-Splat — many thanks for their excellent work.
__device__ inline float2 computeEllipseIntersection(
    const float4 con_o, const float disc, const float t, const float2 p,
    const bool isY, const float coord)
{
    float p_u = isY ? p.y : p.x;
    float p_v = isY ? p.x : p.y;
    float coeff = isY ? con_o.x : con_o.z;

    float h = coord - p_u;  // h = y - p.y for y, x - p.x for x
    float sqrt_term = sqrt(disc * h * h + t * coeff);

    return {
      (-con_o.y * h - sqrt_term) / coeff + p_v,
      (-con_o.y * h + sqrt_term) / coeff + p_v
    };
}


// This is built upon Speedy-Splat — many thanks for their excellent work.
__device__ inline uint32_t processTiles(
    const float4 con_o, const float disc, const float t, const float2 p,
    float2 bbox_min, float2 bbox_max,
    float2 bbox_argmin, float2 bbox_argmax,
    int2 rect_min, int2 rect_max,
    const dim3 grid, const bool isY,
    uint32_t idx, uint32_t off, float depth,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted
    )
{
    const uint32_t off_start =off;

    // ---- AccuTile Code ---- //

    // Set variables based on the isY flag
    float BLOCK_U = isY ? BLOCK_Y : BLOCK_X;
    float BLOCK_V = isY ? BLOCK_X : BLOCK_Y;

    if (isY) {
      rect_min = {rect_min.y, rect_min.x};
      rect_max = {rect_max.y, rect_max.x};

      bbox_min = {bbox_min.y, bbox_min.x};
      bbox_max = {bbox_max.y, bbox_max.x};

      bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
      bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
    }

    uint32_t tiles_count = 0;
    float2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line, max_line;

    // Initialize max line
    // Just need the min to be >= all points on the ellipse
    // and  max to be <= all points on the ellipse
    intersect_max_line = {bbox_max.y, bbox_min.y};

    min_line = rect_min.x * BLOCK_U;
    // Initialize min line intersections.
    if (bbox_min.x <= min_line) {
      // Boundary case
      intersect_min_line = computeEllipseIntersection(
                con_o, disc, t, p, isY, rect_min.x * BLOCK_U);

    } else {
      // Same as max line
      intersect_min_line = intersect_max_line;
    }


    // Loop over either y slices or x slices based on the `isY` flag.
    for (int u = rect_min.x; u < rect_max.x; ++u)
    {
        // Starting from the bottom or left, we will only need to compute
        // intersections at the next line.
        max_line = min_line + BLOCK_U;
        if (max_line <= bbox_max.x) {
          intersect_max_line = computeEllipseIntersection(
                    con_o, disc, t, p, isY, max_line);
        }

        // If the bbox min is in this slice, then it is the minimum
        // ellipse point in this slice. Otherwise, the minimum ellipse
        // point will be the minimum of the intersections of the min/max lines.
        if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
          ellipse_min = bbox_min.y;
        } else {
          ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
        }

        // If the bbox max is in this slice, then it is the maximum
        // ellipse point in this slice. Otherwise, the maximum ellipse
        // point will be the maximum of the intersections of the min/max lines.
        if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
          ellipse_max = bbox_max.y;
        } else {
          ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
        }
        // Convert ellipse_min/ellipse_max to tiles touched
        // First map back to tile coordinates, then subtract.
        int min_tile_v = max(rect_min.y,
            min(rect_max.y, (int)(ellipse_min / BLOCK_V))
            );
        int max_tile_v = min(rect_max.y,
            max(rect_min.y, (int)(ellipse_max / BLOCK_V + 1))
            );

        tiles_count += max_tile_v - min_tile_v;
        // Only update keys array if it exists.
        if (gaussian_keys_unsorted != nullptr) {
          // Loop over tiles and add to keys array
          for (int v = min_tile_v; v < max_tile_v; v++)
          {
            // For each tile that the Gaussian overlaps, emit a
            // key/value pair. The key is |  tile ID  |      depth      |,
            // and the value is the ID of the Gaussian. Sorting the values
            // with this key yields Gaussian IDs in a list, such that they
            // are first sorted by tile and then by depth.
            uint64_t key = isY ?  (u * grid.x + v) : (v * grid.x + u);
            key <<= 32;
            key |= *((uint32_t*)&depth);
            gaussian_keys_unsorted[off] = key;
            gaussian_values_unsorted[off] = idx;
            off++;
            //CHECK_AND_THROW_ERROR_DEVICE(off - off_start <= BLOCK_SIZE, "Error: More tiles processed than expected. This shouldn't happen."); // Sanity check to ensure we don't write out of bounds
          }
        }
        // Max line of this tile slice will be min lin of next tile slice
        intersect_min_line = intersect_max_line;
        min_line = max_line;
    }
    return tiles_count;
}


// This is built upon Speedy-Splat — many thanks for their excellent work.
__device__ inline uint32_t duplicateToTilesTouched(
    const float2 p, const float4 con_o, const dim3 grid, const float mult,
    uint32_t idx, uint32_t off, float depth,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted
    )
{

    //  ---- SNUGBOX Code ---- //

    // Calculate discriminant
    float disc = con_o.y * con_o.y - con_o.x * con_o.z;

    // If ill-formed ellipse, return 0
    if (con_o.x <= 0 || con_o.z <= 0 || disc >= 0) {
        return 0;
    }

    // Threshold: opacity * Gaussian = 1 / 255
    float t = 2.0f * log(con_o.w * 255.0f);
    t = mult * t; // beta in Compact Box

    float x_term = sqrt(-(con_o.y * con_o.y * t) / (disc * con_o.x));
    x_term = (con_o.y < 0) ? x_term : -x_term;
    float y_term = sqrt(-(con_o.y * con_o.y * t) / (disc * con_o.z));
    y_term = (con_o.y < 0) ? y_term : -y_term;

    float2 bbox_argmin = { p.y - y_term, p.x - x_term };
    float2 bbox_argmax = { p.y + y_term, p.x + x_term };

    float2 bbox_min = {
      computeEllipseIntersection(con_o, disc, t, p, true, bbox_argmin.x).x,
      computeEllipseIntersection(con_o, disc, t, p, false, bbox_argmin.y).x
    };
    float2 bbox_max = {
      computeEllipseIntersection(con_o, disc, t, p, true, bbox_argmax.x).y,
      computeEllipseIntersection(con_o, disc, t, p, false, bbox_argmax.y).y
    };

    // Rectangular tile extent of ellipse
    int2 rect_min = {
        max(0, min((int)grid.x, (int)(bbox_min.x / BLOCK_X))),
        max(0, min((int)grid.y, (int)(bbox_min.y / BLOCK_Y)))
    };
    int2 rect_max = {
        max(0, min((int)grid.x, (int)(bbox_max.x / BLOCK_X + 1))),
        max(0, min((int)grid.y, (int)(bbox_max.y / BLOCK_Y + 1)))
    };

    int y_span = rect_max.y - rect_min.y;
    int x_span = rect_max.x - rect_min.x;

    // If no tiles are touched, return 0
    if (y_span * x_span == 0) {
        return 0;
    }

    // If fewer y tiles, loop over y slices else loop over x slices
    bool isY = y_span < x_span;
    return processTiles(
        con_o, disc, t, p,
        bbox_min, bbox_max,
        bbox_argmin, bbox_argmax,
        rect_min, rect_max,
        grid, isY,
        idx, off, depth,
        gaussian_keys_unsorted,
        gaussian_values_unsorted
    );
}

// Forward version of 2D covariance matrix computation
__device__ inline float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ inline void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}


__device__ inline void computeCov3D_conditional(const glm::vec3 scale, const float scale_t, float mod,
		const glm::vec4 rot, const glm::vec4 rot_r, float* cov3D, float3& p_orig,
		float t, const float timestamp, int idx, bool& mask, float& opacity, const float prefilter_var)
{
	// Create scaling matrix
	float dt=timestamp-t;
	glm::mat4 S = glm::mat4(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;
	S[3][3] = mod * scale_t;

	float a = rot.x;
	float b = rot.y;
	float c = rot.z;
	float d = rot.w;

	float p = rot_r.x;
	float q = rot_r.y;
	float r = rot_r.z;
	float s = rot_r.w;

	// glm::mat4 M_l = glm::mat4(
	// 	a, -b, -c, -d,
	// 	b, a,-d, c,
	// 	c, d, a,-b,
	// 	d,-c, b, a
	// );

	// glm::mat4 M_r = glm::mat4(
	// 	p, q, r, s,
	// 	-q, p,-s, r,
	// 	-r, s, p,-q,
	// 	-s,-r, q, p
	// );
	
	glm::mat4 M_l = glm::mat4(
		 a,  b, -c,  d,
		-b,  a,  d,  c,
		 c, -d,  a,  b,
		-d, -c, -b,  a
	);

	glm::mat4 M_r = glm::mat4(
		p,  q, -r, -s,
		-q, p,  s, -r,
		r, -s,  p, -q,
		s,  r,  q,  p
	);
	// glm stores in column major
	glm::mat4 R = M_r * M_l;
	glm::mat4 M = S * R;
	glm::mat4 Sigma = glm::transpose(M) * M;
	float cov_t = Sigma[3][3];
	float marginal_t = __expf(-0.5*dt*dt/((prefilter_var > 0.0) ? (prefilter_var + cov_t) : cov_t));
	mask = marginal_t > 0.05;
	if (!mask) return;
	opacity*=marginal_t;;
	glm::mat3 cov11 = glm::mat3(Sigma);
	glm::vec3 cov12 = glm::vec3(Sigma[0][3],Sigma[1][3],Sigma[2][3]);
	glm::mat3 cov3D_condition = cov11 - glm::outerProduct(cov12, cov12) / cov_t;

	// Covariance is symmetric, only store upper right
	cov3D[0] = cov3D_condition[0][0];
	cov3D[1] = cov3D_condition[0][1];
	cov3D[2] = cov3D_condition[0][2];
	cov3D[3] = cov3D_condition[1][1];
	cov3D[4] = cov3D_condition[1][2];
	cov3D[5] = cov3D_condition[2][2];
    glm::vec3 delta_mean = cov12 / cov_t * dt;
	p_orig.x+=delta_mean.x;
	p_orig.y+=delta_mean.y;
	p_orig.z+=delta_mean.z;
}



#endif