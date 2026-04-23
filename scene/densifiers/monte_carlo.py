class MontecarloPruner:
    def __init__(self, opt):
        super().__init__(opt, "montecarlo")

    def training_setup(self, gaussians,reset_accumulated_gradients = True):
        pass

    def densify_and_prune(self, iteration, scene, gaussians, radii, pipe, bg, *, prune_only):
        num_cams = 40
        my_viewpoint_stack = scene.getTrainCameras()            
        camlist = sampling_cameras(my_viewpoint_stack, num_cams,dimensions=4)

#        error_offsets = torch.zeros((gaussians.get_xyz.shape[0],), device=gaussians.get_xyz.device)


#        drop_percentage = 10
        for view in range(len(camlist)):
            gt_image,viewpoint_cam = camlist[view]
            viewpoint_cam=viewpoint_cam.cuda()
            gt_image=gt_image.detach()

            render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, fastgs_mult)
            render_image = render_pkg["render"]
            l1_norm = _get_loss_map(render_image, gt_image)
            print(gaussians._opacities.grad)





    def add_densification_stats_grad(self, *,gaussians, iteration, viewspace_point_grad, update_filter, radii,avg_t_gradient):
        pass



    def per_iteration(self, iteration, scene, gaussians, radii, pipe, bg):
        pass

    def get_save_vars(self,gaussians):
        """Return variables that should be saved or loaded with the scene, as attribute names """
        return []

    def densification_postfix(self, gaussians):
        pass

    def prune_points(self, prune_mask):
        pass

    @torch.no_grad()
    def apply_debug_colour(self, gaussians, scene,pipe,bg, debug_type=""):
        pass
