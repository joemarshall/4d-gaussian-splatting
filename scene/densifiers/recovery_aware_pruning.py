import torch
from .densifier_base import DensifierBase   

class RecoveryAwarePruner(DensifierBase):
    def densify_and_prune(self, iteration, scene, gaussians, radii,pipe, bg):
        # we do nothing on densification iterations
        pass

    def add_densification_stats_grad(self,*,gaussians,iteration,viewspace_point_grad,update_filter, radii,avg_t_gradient):
        pass

    def get_save_vars(self,gaussians):
        return []
    
    def per_iteration(self, iteration, scene, gaussians, radii, pipe, bg):
        prune_after = getattr(self.options, 'recovery_aware_pruning_iter_offset', 30)
        offset_iter = iteration - prune_after
        #print("Recovery-aware pruning: iteration {}, offset iteration {}".format(iteration, offset_iter))
        if offset_iter >= self.options.densify_from_iter and offset_iter < self.options.densify_until_iter and offset_iter % self.options.opacity_reset_interval == 0:
            # prune everything less than 0.05 opacity
            print("Recovery-aware pruning: removing points with opacity less than 0.05 at iteration {}".format(iteration))
            prune_mask = (gaussians.get_opacity < 0.05).squeeze()
            gaussians.prune_points(prune_mask)


                         



