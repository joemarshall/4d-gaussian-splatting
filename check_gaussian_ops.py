import torch


from gaussian_renderer.diff_gaussian_rasterization import C_rasterize_gaussians


P = 100
W = 64
H = 64




fw_args = (
    torch.zeros(3),
    torch.zeros((P,3)),
    torch.zeros((P,3)),
    torch.zeros((1,H,W)),
    torch.zeros((P,1)), // opacities
    torch.zeros((P,1)),// times
    torch.zeros((P,3)), // scales
    torch.zeros((P,1)), // scales_t
    torch.zeros((P,4)), // rotations 
    torch.zeros((P,4)), // rotations_r
    1.0, // scale_modifier
    torch.zeros((P,6)), 
    0.0,
    torch.eye(4),
    torch.eye(4),
    1.0,
    1.0,
    64,
    64,
    torch.zeros((P, 3 * (degree + 1) ** 2)),
    3,
    3,
    torch.zeros((3,)),
    0.0,
    1.0,
    True,
    3,
    False,
    False,
    False,
    torch.zeros((W*H)),
    
)


torch.library.opcheck(C_rasterize_gaussians,fw_args)
