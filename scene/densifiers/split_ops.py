import torch


from utils.general_utils import inverse_sigmoid,build_rotation, build_rotation_4d

def densify_and_split(gaussians, selected_pts_mask, N=2):
    n_init_points = gaussians.get_xyz.shape[0]

    new_scaling = gaussians.scaling_inverse_activation(gaussians.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
    new_rotation = gaussians._rotation[selected_pts_mask].repeat(N,1)
    new_features_dc = gaussians._features_dc[selected_pts_mask].repeat(N,1,1)
    new_features_rest = gaussians._features_rest[selected_pts_mask].repeat(N,1,1)
    new_opacity = gaussians._opacity[selected_pts_mask].repeat(N,1)
    
    if not gaussians.rot_4d:
        stds = gaussians.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(gaussians._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussians.get_xyz[selected_pts_mask].repeat(N, 1)
        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        if gaussians.gaussian_dim == 4:
            stds_t = gaussians.get_scaling_t[selected_pts_mask].repeat(N,1)
            means_t = torch.zeros((stds_t.size(0), 1),device="cuda")
            samples_t = torch.normal(mean=means_t, std=stds_t)
            new_t = samples_t + gaussians.get_t[selected_pts_mask].repeat(N, 1)
            new_scaling_t = gaussians.scaling_inverse_activation(gaussians.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
    else:
        stds = gaussians.get_scaling_xyzt[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 4),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation_4d(gaussians._rotation[selected_pts_mask], gaussians._rotation_r[selected_pts_mask]).repeat(N,1,1)
        new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussians.get_xyzt[selected_pts_mask].repeat(N, 1)
        new_xyz = new_xyzt[...,0:3]
        new_t = new_xyzt[...,3:4]
        new_scaling_t = gaussians.scaling_inverse_activation(gaussians.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation_r = gaussians._rotation_r[selected_pts_mask].repeat(N,1)

    gaussians.densification_postfix(
        new_xyz, new_features_dc, new_features_rest,
        new_opacity, new_scaling, new_rotation,
        new_t, new_scaling_t, new_rotation_r,
    )

    # get rid of original points that were just split
    prune_filter = torch.cat((
        selected_pts_mask,
        torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=torch.bool),
    ))
    gaussians.prune_points(prune_filter)

def densify_and_clone(gaussians,selected_pts_mask,N=2):
    """ Clone points at selected_pts_mask, without any perturbation (i.e. same position, features etc). This is used for
    things which have enough gradient that they will end up different to the initial points after next optimization step."""
#    print("densify and clone:",selected_pts_mask.sum(),"/",len(selected_pts_mask))
    new_xyz = gaussians._xyz[selected_pts_mask].repeat(N-1,1)
    new_features_dc = gaussians._features_dc[selected_pts_mask].repeat(N-1,1,1)
    new_features_rest = gaussians._features_rest[selected_pts_mask].repeat(N-1,1,1)
    new_opacities = gaussians._opacity[selected_pts_mask].repeat(N-1,1)
    new_scaling = gaussians._scaling[selected_pts_mask].repeat(N-1,1)
    new_rotation = gaussians._rotation[selected_pts_mask].repeat(N-1,1)
    new_t = None
    new_scaling_t = None
    new_rotation_r = None
    if gaussians.gaussian_dim == 4:
        new_t = gaussians._t[selected_pts_mask].repeat(N-1,1)
        new_scaling_t = gaussians._scaling_t[selected_pts_mask].repeat(N-1,1)
        if gaussians.rot_4d:
            new_rotation_r = gaussians._rotation_r[selected_pts_mask].repeat(N-1,1)

    gaussians.densification_postfix(
        new_xyz, new_features_dc, new_features_rest,
        new_opacities, new_scaling, new_rotation,
        new_t, new_scaling_t, new_rotation_r,
    )


def densify_and_split_long_axis(gaussians, selected_pts_mask, rate=1.5, N=2):
    assert( N==2) # long axis split only implemented for 2x splits for now

    # non geometric things
    new_rotation = gaussians._rotation[selected_pts_mask].repeat(N, 1)
    new_features_dc = gaussians._features_dc[selected_pts_mask].repeat(N, 1, 1)
    new_features_rest = gaussians._features_rest[selected_pts_mask].repeat(N, 1, 1)
    new_opacity = inverse_sigmoid(gaussians.get_opacity[selected_pts_mask] * 0.6).repeat(N, 1)


    if gaussians.rot_4d:
        # rotation in x,y,z and t offset on largest axis
        stds = gaussians.get_scaling_xyzt[selected_pts_mask]
        _max_values, max_indices = torch.max(stds, dim=1, keepdim=True)


        mask = torch.zeros_like(stds, dtype=torch.bool).scatter_(1, max_indices, True)
        samples = stds * mask * 1.5
        offsets = torch.linspace(-1.0,1.0,steps = N,device = mask.device,dtype = stds.dtype)
        
#        samples = torch.cat([samples, -samples], dim=0)
        # this could probably be done with bmm, but this is clearer and N is small
        samples = torch.cat([offsets[x] * samples for x in range(N)], dim=0)

        rots = build_rotation_4d(gaussians._rotation[selected_pts_mask], gaussians._rotation_r[selected_pts_mask]).repeat(N, 1, 1)

        new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussians.get_xyzt[selected_pts_mask].repeat(N, 1)
        new_xyz = new_xyzt[...,0:3]
        new_t = new_xyzt[...,3:4]
        new_scaling_xyzt = gaussians.scaling_inverse_activation(gaussians.get_scaling_xyzt[selected_pts_mask].repeat(N,1) / (0.8*N))
        
        new_scaling_t = new_scaling_xyzt[...,3:4]
        new_scaling = new_scaling_xyzt[...,0:3]

        new_scaling_t = gaussians.scaling_inverse_activation(gaussians.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation_r = gaussians._rotation_r[selected_pts_mask].repeat(N,1)

    else:
        # rotation in just x,y,z offset on largest axis
        stds = gaussians.get_scaling[selected_pts_mask]
        _max_values, max_indices = torch.max(stds, dim=1, keepdim=True)
        mask = torch.zeros_like(stds, dtype=torch.bool).scatter_(1, max_indices, True)
        samples = stds * mask * 1.5

        offsets = torch.linspace(-1.0,1.0,steps = N,device = mask.device,dtype = stds.dtype)
#        samples = torch.cat([samples, -samples], dim=0)
        # this could probably be done with bmm, but this is clearer and N is small
        samples = torch.cat([offsets[x] * samples for x in range(N)], dim=0)
        
        rots = build_rotation(gaussians._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_scaling = gaussians.scaling_inverse_activation(gaussians.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussians.get_xyz[selected_pts_mask].repeat(N, 1)
        new_t = None
        new_scaling_t = None
        new_rotation_r = None

        if gaussians.gaussian_dim == 4:        
            stds_t = gaussians.get_scaling_t[selected_pts_mask].repeat(N,1)
            means_t = torch.zeros((stds_t.size(0), 1),device="cuda")
            samples_t = torch.normal(mean=means_t, std=stds_t)
            new_t = samples_t + gaussians.get_t[selected_pts_mask].repeat(N, 1)
            new_scaling_t = gaussians.scaling_inverse_activation(gaussians.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))


    gaussians.densification_postfix(
        new_xyz, new_features_dc, new_features_rest,
        new_opacity, new_scaling, new_rotation,
        new_t, new_scaling_t, new_rotation_r,
    )

    total_points = gaussians.get_xyz.shape[0]
    # get rid of original points that we just split
    prune_filter = torch.cat((selected_pts_mask, torch.zeros(total_points - selected_pts_mask.shape[0], device="cuda", dtype=bool) ))
    gaussians.prune_points(prune_filter)

def clone_split_prune(gaussians, clones=None,splits=None,prunes=None,*,repeat_count = 2 ,long_axis_split=False):
    initial_point_count = gaussians.get_xyz.shape[0]
    # don't split or clone prunes, because they will be removed next time anyway
    # (as opacity never goes up after split/clone)
    if prunes is not None:
        if splits is not None:
            splits[prunes] = False
        if clones is not None:
            clones[prunes] = False
    if long_axis_split:
        # long axis split only
        if clones is not None:
            splits = torch.logical_or(splits, clones)
        clones = None
    # split removes the original split points, so we need to remake the prune mask
    # taking that into account
    if prunes is not None and splits is not None:
        prunes = prunes [~splits]

    if clones is not None:
        densify_and_clone(gaussians, clones, repeat_count)

    # clone just extends the points so 
    # we just need to make the other tensors bigger
    if splits is not None:
        if long_axis_split:
            densify_and_split_long_axis(gaussians,splits,repeat_count)
        else:
            splits = torch.cat([splits, torch.zeros((gaussians.get_xyz.shape[0] - splits.shape[0]), device=splits.device, dtype=torch.bool)])
            densify_and_split(gaussians, splits, repeat_count)
    if prunes is not None:
        # if we have cloned or split, then a whole
        # load of new points have been added, so we need to resize the prunes maks to include it
        prunes = torch.cat([prunes, torch.zeros((gaussians.get_xyz.shape[0] - prunes.shape[0]), device=prunes.device, dtype=torch.bool)])
        gaussians.prune_points(prunes)
    points_left = gaussians.get_xyz.shape[0]
    if long_axis_split:
        print(f"\nclone_split_prune ({repeat_count}x): {splits.sum().item() if splits is not None else 0} long_axis_splits, {prunes.sum().item() if prunes is not None else 0} prunes n={points_left}({initial_point_count})")
    else:
        print(f"\nclone_split_prune ({repeat_count}x):{clones.sum().item() if clones is not None else 0} clones, {splits.sum().item() if splits is not None else 0} splits, {prunes.sum().item() if prunes is not None else 0} prunes n={points_left}({initial_point_count})")
    mean_opacity = torch.mean(gaussians.get_opacity)
    mean_scaling = torch.mean(gaussians.get_scaling)
    if gaussians.gaussian_dim == 4:

        cov_t = gaussians.get_cov_t()
        mean_cov_t = torch.mean(cov_t) 
        mean_cov_t = torch.sqrt(mean_cov_t)
        log_bins = torch.logspace(-4, 5, steps=10)
        scaling_histogram = torch.histogram(cov_t.detach().cpu(),bins=log_bins)[0].tolist()
    else:
         mean_cov_t = None
         scaling_histogram = ""

    mean_dc = torch.mean(gaussians.get_sh_features_dc)
    print(f"mean_opacity: {mean_opacity}, mean_scaling: {mean_scaling}, mean_cov_t: {mean_cov_t}")
    print(f"mean_dc: {mean_dc}")
    print("Time scaling histogram:",scaling_histogram)
