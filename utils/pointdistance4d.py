# at each timestep we have:
# 1) points which are centred at this timestamp
# 2) points which are live at this timestamp
#
# Do NN from centred points to live points to get distance to nearest live point for
# each centred point
# and set radius of that point based on that distance
#
# n.b. this is O(N^2) sort of, but time filtered

import torch


# first, for each axis, spread out the number so that
# it has two zeros in between each bit
def spread_bits(x):
    """
    Spread 21 bits -> 63 bits by putting two zeros in between each bit"""
    orig = x
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8)) & 0x100F00F00F00F00F
    x = (x | (x << 4)) & 0x10C30C30C30C30C3
    x = (x | (x << 2)) & 0x1249249249249249
    # for k in range(x.shape[0]):
    #     original_with_spaces = "{0:b}".format(orig[k].item())
    #     original_with_spaces = "".join([c+"  " for c in original_with_spaces])
    #     x_with_zeros = "{0:b}".format(x[k].item())
    #     print("Spread\n",original_with_spaces,"\n",x_with_zeros)
    return x


def _create_interleaved_index(points: torch.Tensor):
    """
    points: (N, 3) tensor of point positions

    Returns:
        indices (N,) tensor of interleaved indices for each point
    """
    morton_index = (
        spread_bits(points[:, 0])
        | (spread_bits(points[:, 1]) << 1)
        | (spread_bits(points[:, 2]) << 2)
    )
    source_indices = torch.arange(
        points.shape[0], device=points.device, dtype=torch.int64
    )
    morton_index = torch.stack([morton_index, source_indices], dim=1)

    # index_array looks like [[source_index, morton_index, morton_rank]] for each point
    # sort by morton_index to get the order of points in morton order
    # and use the indices returned by torch.sort to add a column to index_array
    # which is the index of the point in the sorted morton array
    sort_order = torch.argsort(morton_index[:, 0])
    morton_index = morton_index[sort_order]
    morton_rank = torch.empty_like(sort_order)
    morton_rank[sort_order] = torch.arange(
        points.shape[0], device=points.device, dtype=torch.int64
    )
    index_array = sort_order

    return morton_index, index_array


def pointDistance4D(
    points: torch.Tensor,
    start_times: torch.Tensor,
    durations: torch.Tensor,
    timestamps: torch.Tensor,
):
    """
    points: (N, 3) tensor of point positions
    start_times: (N,) tensor of start times for each point
    durations: (N,) tensor of durations for each point
    timestamps: (M,) tensor of timestamps to calculate distances at

    Returns:
        distances (N,) tensor of the distance to the nearest point that is live at the centre timestamp
            for each point.
    """
    # Calculate end times for each point
    distances = torch.zeros(points.shape[0], device=points.device)

    max_pos = torch.max(points, dim=0).values
    min_pos = torch.min(points, dim=0).values
    print("Point cloud bounds: min {}, max {}".format(min_pos, max_pos))
    index_divisor = ((max_pos - min_pos)+1e-12) / 1048576
    print("DIVISOR:",index_divisor)

    # 3d indices in 21 bit range for each point
    index_points = torch.floor((points - min_pos) / index_divisor).long()
    # now take bitwise selections from index_points to get a single 60 bit integer for each point
    # which is made of interlaced bits from each of the dimensions

    morton_sorted_indices, index_to_morton_rank = _create_interleaved_index(
        index_points
    )



    # now reshape everything to be morton-sorted
    reshape_index = index_to_morton_rank
    unreshape_index = torch.argsort(reshape_index).squeeze()

    start_times = start_times[reshape_index]
    durations = durations[reshape_index]
    points = points[reshape_index]

    max_pos = torch.max(points, dim=0).values
    min_pos = torch.min(points, dim=0).values
    print("Point cloud bounds: min {}, max {}".format(min_pos, max_pos))
    index_divisor = ((max_pos - min_pos)+1e-12) / 2097151

    # 3d indices in 21 bit range for each point
    index_points = torch.floor((points - min_pos) / index_divisor).long()


    last_mi = None

    diffs = torch.linalg.vector_norm(points[1:] - points[:-1], dim=1)

    for i in range(points.shape[0]):
        x = index_points[i, 0].item()
        y = index_points[i, 1].item()
        z = index_points[i, 2].item()
        morton_index = (
            spread_bits(x)
            | (spread_bits(y) << 1)
            | (spread_bits(z) << 2)
        )
        if i<diffs.shape[0]:
            if diffs[i] > 3.0:
                print("Distance between {} and {}: {}".format(i,i+1,diffs[i].item()))
                print("Morton vals:",morton_index, last_mi)
                print("Points",points[i:i+2])
                print("index points:",index_points[i:i+2])
        last_mi = morton_index


    end_times = start_times + durations
    centre_times = start_times + durations / 2.0

    last_t = timestamps[0]-(timestamps[1]-timestamps[0])
    for t in timestamps:
        # centre points are those which are centred at this timestamp
        # i.e. only one timestamp per point
        centre_mask = torch.logical_and(
            centre_times > last_t, centre_times <= t
        ).squeeze()
        # live points are those which are visible at this timestamp
        # i.e. multiple timestamps per point
        live_mask = torch.logical_and(start_times <= t, end_times >= t).squeeze()
        last_t = t
        if centre_mask.sum() > 0 and live_mask.sum() > 0:
            print(
                "Calculating centre and live masks for timestamp {}: {} centre points, {} live points".format(
                    t, centre_mask.sum().item(), live_mask.sum().item()
                )
            )

            # everything is morton sorted, so we just need to iterate through centre points in order
            # and use prev and next live points to calculate distance to nearest live point
            centre_point_indices = centre_mask.nonzero().squeeze(-1)
            live_point_indices = live_mask.nonzero().squeeze(-1)

            #print(centre_point_indices,live_point_indices,centre_point_indices.shape, live_point_indices.shape)
            centre_points_in_live_points = torch.searchsorted(live_point_indices, centre_point_indices)

            offset_distances = torch.zeros(6, centre_point_indices.shape[0], device=points.device)

            for x in torch.arange(-3,4):
                if x<0:
                    offset_pos = torch.clamp(centre_points_in_live_points+x,0,live_point_indices.shape[0]-1)
                    distance_vals = torch.linalg.norm(points[centre_point_indices] - points[live_point_indices[offset_pos]], dim=1)
                    offset_distances[x+3] = distance_vals
                elif x>0:
                    offset_pos = torch.clamp(centre_points_in_live_points+x,0,live_point_indices.shape[0]-1)
                    distance_vals = torch.linalg.norm(points[centre_point_indices] - points[live_point_indices[offset_pos]], dim=1)
                    offset_distances[x+2] = distance_vals
                else:
                    pass

            offset_distances[offset_distances==0.0]=9999
                

            distances[centre_point_indices] = torch.min(offset_distances, dim=0).values
            # for x in centre_point_indices:
            #     for y in centre_point_indices:
            #         if x<y:
            #             d = torch.linalg.norm(points[x:x+1] - points[y:y+1],dim=1)
            #             #print("Distance between {} and {}: {}".format(x,y,d))
            #             if d < distances[x]:
            #                 print("Found a closer point for {}: {} < {}".format(x,d,distances[x]))


    print("Unreshaping distances back to original order",unreshape_index.shape, distances.shape)
    # now reorder back to original order not morton sorted order
    distances = distances[unreshape_index]
    print("Done")

    return distances


# find nearest points in 2d (i.e. in camera image space usually) for each of the query points
def find2d_nearest_points(points2d: torch.tensor, query_points2d: torch.tensor):
    """
    points2d: (N, 2) tensor of 2D point positions
    query_points2d: (M, 2) tensor of 2D query point positions

    Returns:
        nearest_indices: (M,) tensor of indices of the nearest point in points2d for each query point
        nearest_distances: (M,) tensor of distances to the nearest point in points2d for each query point
    """
    q = query_points2d.to(dtype=torch.float32)
    p = points2d.to(dtype=torch.float32)

    # Build 2D Morton index for points2d
    # Quantize p and q to 21-bit integers along each axis
    all_pts = torch.cat([p, q], dim=0)
    min_pos = all_pts.min(dim=0).values
    max_pos = all_pts.max(dim=0).values
    index_divisor = ((max_pos - min_pos) + 1e-12) / 1048576.0

    p_idx = torch.floor((p - min_pos) / index_divisor).long()  # (N, 2)
    q_idx = torch.floor((q - min_pos) / index_divisor).long()  # (M, 2)

    # Interleave bits for 2D Morton code: bits from x and y alternated
    def spread_bits_2d(x):
        x = x & 0x3FFFFFFF                  # keep 30 bits
        x = (x | (x << 16)) & 0x0000FFFF0000FFFF
        x = (x | (x << 8))  & 0x00FF00FF00FF00FF
        x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0F
        x = (x | (x << 2))  & 0x3333333333333333
        x = (x | (x << 1))  & 0x5555555555555555
        return x

    p_morton = spread_bits_2d(p_idx[:, 0]) | (spread_bits_2d(p_idx[:, 1]) << 1)  # (N,)
    q_morton = spread_bits_2d(q_idx[:, 0]) | (spread_bits_2d(q_idx[:, 1]) << 1)  # (M,)

    # Sort p by Morton code
    p_sort_order = torch.argsort(p_morton)          # (N,) sorted indices into p
    p_morton_sorted = p_morton[p_sort_order]        # (N,) sorted Morton codes
    p_sorted = p[p_sort_order]                      # (N, 2) sorted positions

    # For each query point, find its position in the sorted Morton order via searchsorted
    q_insert_pos = torch.searchsorted(p_morton_sorted, q_morton)  # (M,)

    # Check a window of neighbours around the insertion position in Morton order
    window = 32
    N = p_sorted.shape[0]
    M = q.shape[0]

    offsets = torch.arange(-window, window + 1, device=q.device)           # (W,)
    # candidate indices into p_sorted for each query: (M, W)
    cand_pos = (q_insert_pos.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, N - 1)

    # Gather candidate positions
    cand_pts = p_sorted[cand_pos.reshape(-1)].reshape(M, 2 * window + 1, 2)  # (M, W, 2)

    # Compute squared distances from each query to its candidates
    diff = cand_pts - q.unsqueeze(1)                                          # (M, W, 2)
    dist2 = (diff * diff).sum(dim=2)                                          # (M, W)

    # Exclude exact overlaps
    dist2[dist2 == 0.0] = torch.inf

    # Find nearest candidate
    min_dist2, min_w_idx = torch.min(dist2, dim=1)                            # (M,)
    nearest_distances = torch.sqrt(min_dist2)

    # Map back from sorted p index to original p index
    nearest_indices = p_sort_order[cand_pos[torch.arange(M, device=q.device), min_w_idx]]

    return nearest_indices, nearest_distances

if __name__ == "__main__":
    import torch

    points = torch.rand((1048576,2),device="cuda")*100.0 
    points = points.to(dtype=torch.float32)
    find2d_nearest_points(points,points[0:262144])


    # import torch,numpy as np

    # min_pos=torch.tensor([-9.6568, -9.9596, -8.6377], device='cuda:0')
    # max_pos=torch.tensor([16.7596,  5.4903, 22.7297], device='cuda:0')
    # index_divisor = ((max_pos - min_pos)+1e-12) / 1048576



    # points_raw=torch.tensor([[ 9.9712, -4.2354, -0.8641],
    #     [ 3.7196, -4.1225, -4.4853]], device='cuda:0')

    # points = torch.floor((points_raw - min_pos) / index_divisor).long()


    # morton_index = (
    #     torch.bitwise_and(torch.bitwise_and(spread_bits(points[:, 0]), (spread_bits(points[:, 1]) << 1)), (spread_bits(points[:, 2]) << 2))
    # )
    # print(points)
    # print(morton_index)


    # import sys
    # sys.exit(0)



    # import torch
    # with torch.no_grad():
    #     gdata,_,_ = torch.load("output\\9moving\\model_output\\chkpnt_resume.pth",map_location="cuda",weights_only=False)
    #     points = gdata["_xyz"]


    #     # test the function with a simple example
    #     #points= torch.rand((24,3),device="cuda")
    # #    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    #     start_times = torch.zeros(points.shape[0],device="cuda")
    #     durations = torch.ones_like(start_times) * 10.0
    #     timestamps = torch.tensor([0.0,5.0],device="cuda")
    #     distances = pointDistance4D(points, start_times, durations, timestamps)
    #     print(distances)
        
    #     from simple_knn._C import distCUDA2
    #     dist2 =    distCUDA2(points)
    #     print("!!!!!!!!!!!!!!!!!")
    #     print(dist2)

    # dist3 = torch.zeros(points.shape[0],device="cuda")
    # for x in range(points.shape[0]):
    #     diffs = torch.linalg.norm(points[x] - points,dim=1)
    #     diffs = diffs[diffs>0.0]
    #     min_dist = torch.min(diffs)
    #     dist3[x]=min_dist
    
    # print(dist3)



