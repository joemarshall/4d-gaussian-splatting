import torch
from .densifier_base import DensifierBase   
from .split_ops import *

class OpacityReset(DensifierBase):
    def __init__(self,opt):
        super().__init__(opt, "opacity_reset")

    def densify_and_prune(self, iteration, scene, gaussians, radii,pipe, bg, *, options):
        gaussians.reset_opacity()

    def add_densification_stats_grad(self,*,gaussians,iteration,viewspace_point_grad,update_filter, radii,avg_t_gradient):
        pass

    def get_save_vars(self,gaussians):
        return []

    def needs_densification_or_pruning(self, gaussians, iteration):
        densify_from_iter = self._get_option("densify_from_iter", 0)
        densify_until_iter = self._get_option("densify_until_iter", 1e9)

        reset_at = self._get_option('opacity_reset_interval', 3000)
        if iteration>densify_from_iter and iteration < densify_until_iter and iteration % reset_at ==0:
            return {"densify": False, "prune": True, "final_prune": False}
        else:
            return None
    


                         



