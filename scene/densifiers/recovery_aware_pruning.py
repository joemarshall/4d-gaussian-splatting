import torch
from .densifier_base import DensifierBase   
from .split_ops import *

class RecoveryAwarePruner(DensifierBase):
    def __init__(self,opt):
        super().__init__(opt, "recovery_aware")

    def densify_and_prune(self, iteration, scene, gaussians, radii,pipe, bg, *, options):
        # only gets called if needs_densification is true
        prune_opacity = getattr(self.options, 'prune_opacity_threshold`', 0.05)
        # prune everything less than 0.01 opacity
        print("Recovery-aware pruning: removing points with opacity less than {} at iteration {}".format(prune_opacity, iteration))
        prune_mask = (gaussians.get_opacity < prune_opacity).squeeze()
        clone_split_prune(gaussians,None,None,prune_mask)

    def add_densification_stats_grad(self,*,gaussians,iteration,viewspace_point_grad,update_filter, radii,avg_t_gradient):
        pass

    def get_save_vars(self,gaussians):
        return []

    def needs_densification_or_pruning(self, gaussians, iteration):
        reset_interval = self._get_option('opacity_reset_interval', 3000)

        prune_after = self._get_option('recovery_aware_pruning_iter_offset', 50)
        densify_from_iter = self._get_option("densify_from_iter", 0)
        densify_until_iter = self._get_option("densify_until_iter", 1e9)

        offset_iter = iteration - prune_after
        if offset_iter >= densify_from_iter and offset_iter < densify_until_iter and offset_iter % reset_interval == 0:
            return {"densify": False, "prune": True, "final_prune": False}
        else:
            return None
    


                         



