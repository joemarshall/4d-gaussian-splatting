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

def densify_and_clone(gaussians,selected_pts_mask):
    """ Clone points at selected_pts_mask, without any perturbation (i.e. same position, features etc). This is used for
    things which have enough gradient that they will end up different to the initial points after next optimization step."""
    print("densify and clone:",selected_pts_mask.sum(),"/",len(selected_pts_mask))
    new_xyz = gaussians._xyz[selected_pts_mask]
    new_features_dc = gaussians._features_dc[selected_pts_mask]
    new_features_rest = gaussians._features_rest[selected_pts_mask]
    new_opacities = gaussians._opacity[selected_pts_mask]
    new_scaling = gaussians._scaling[selected_pts_mask]
    new_rotation = gaussians._rotation[selected_pts_mask]
    new_t = None
    new_scaling_t = None
    new_rotation_r = None
    if gaussians.gaussian_dim == 4:
        new_t = gaussians._t[selected_pts_mask]
        new_scaling_t = gaussians._scaling_t[selected_pts_mask]
        if gaussians.rot_4d:
            new_rotation_r = gaussians._rotation_r[selected_pts_mask]

    gaussians.densification_postfix(
        new_xyz, new_features_dc, new_features_rest,
        new_opacities, new_scaling, new_rotation,
        new_t, new_scaling_t, new_rotation_r,
    )


def densify_and_split_long_axis(gaussians, selected_pts_mask, rate=1.5):

    # non geometric things
    new_rotation = gaussians._rotation[selected_pts_mask].repeat(2, 1)
    new_features_dc = gaussians._features_dc[selected_pts_mask].repeat(2, 1, 1)
    new_features_rest = gaussians._features_rest[selected_pts_mask].repeat(2, 1, 1)
    new_opacity = inverse_sigmoid(gaussians.get_opacity[selected_pts_mask] * 0.6).repeat(2, 1)

    if gaussians.rot_4d:
        # rotation in x,y,z and t offset on largest axis
        stds = gaussians.get_scaling_xyzt[selected_pts_mask]
        max_values, max_indices = torch.max(stds, dim=1, keepdim=True)
        mask = torch.zeros_like(stds, dtype=torch.bool).scatter_(1, max_indices, True)
        samples = stds * mask * 1.5
        samples = torch.cat([samples, -samples], dim=0)

        rots = build_rotation_4d(gaussians._rotation[selected_pts_mask], gaussians._rotation_r[selected_pts_mask]).repeat(2, 1, 1)

        new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussians.get_xyzt[selected_pts_mask].repeat(2, 1)
        new_xyz = new_xyzt[...,0:3]
        new_t = new_xyzt[...,3:4]

        new_scaling_t = gaussians.scaling_inverse_activation(gaussians.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation_r = gaussians._rotation_r[selected_pts_mask].repeat(N,1)

    else:
        # rotation in just x,y,z offset on largest axis
        stds = gaussians.get_scaling[selected_pts_mask]
        max_values, max_indices = torch.max(stds, dim=1, keepdim=True)
        mask = torch.zeros_like(stds, dtype=torch.bool).scatter_(1, max_indices, True)
        samples = stds * mask * 1.5
        samples = torch.cat([samples, -samples], dim=0)
        rots = build_rotation(gaussians._rotation[selected_pts_mask]).repeat(2, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussians.get_xyz[selected_pts_mask].repeat(2, 1)
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

    # get rid of original points that we just split
    prune_filter = torch.cat((selected_pts_mask, torch.zeros(2 * selected_pts_mask.sum(), device="cuda", dtype=bool)))
    gaussians.prune_points(prune_filter)