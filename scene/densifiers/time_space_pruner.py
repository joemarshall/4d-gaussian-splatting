import torch
from .densifier_base import DensifierBase   
from .split_ops import *

from gaussian_renderer import calculate_gaussian_contribution

# prune by time/space score -
# Spatial-Temporal Variation Score from https://4dgs-1k.github.io/
class TimeSpacePruner(DensifierBase):
    def __init__(self,opt):
        super().__init__(opt, "time_space_pruning")

    def compute_temporal_scores(self, gaussians,timestamp):
        """Compute per-Gaussian temporal score (Eq. 5-6 from arXiv 2503.16422).

        The temporal score measures how much each 4D Gaussian contributes 
        at this timestamp, irrespective of gaussian opacity (which is handled by the
        spatial score)

        Args:
            timestamp: float T

        Returns:
            Tensor of shape (N,) with temporal scores in [0, 1].
            Returns a tensor of ones when gaussian_dim != 4.
        """
        if gaussians.gaussian_dim != 4:
            return torch.ones(gaussians.get_xyz.shape[0], device="cuda")
        with torch.no_grad():
            sigma = gaussians.get_marginal_t(timestamp)
            return sigma


    def densify_and_prune(self, iteration, scene, gaussians, radii,pipe, bg, *, options):
        # render *all* training views
        # for each gaussian, calculate time/space score
        # based on:
        #
        #
        # 1) count of frames including the gaussian (time score)
        # 2) visibile pixels for the gaussian (space score)

        all_views = scene.getTrainCameras()
        timestamps = all_views.get_timestamps()
        cameras = all_views.get_num_different_cameras()

        tensor_device = gaussians.get_xyz.device



        all_views.set_names_only(True)
        num_points = gaussians.get_xyz.shape[0]
        frame_counts = torch.zeros(num_points,dtype=torch.int32,device=tensor_device)

        st_sums = torch.zeros(num_points,device=tensor_device)

        for t in timestamps:
            print("Processing timestamp", t)
            spatial_sum = torch.zeros(num_points,device=tensor_device)
            found_frame = torch.zeros(num_points,dtype=torch.int32,device=tensor_device)
            indices = all_views.get_indices_for_timestamp(t)
            for i in indices:
                cam = all_views[i][1].cuda()
                error_map = torch.zeros((cam.image_height*cam.image_width),device=tensor_device)
                outputs = calculate_gaussian_contribution(cam, gaussians, pipe, bg, error_map = error_map)
                contributions = outputs["visual_contribution"]
                found_frame[contributions>0]+=1
                spatial_sum += contributions
            spatial_sum = spatial_sum / (spatial_sum.max() + 1e-8)
            temporal_score = self.compute_temporal_scores(gaussians,t)
#            print(torch.min(temporal_score).item(), torch.max(temporal_score).item())
#            print(torch.min(spatial_sum).item(), torch.max(spatial_sum).item())
            st_sums+= spatial_sum* temporal_score.squeeze()

            frame_counts[found_frame>0]+=1
        # first get rid of invisibles
        invisibles = frame_counts == 0

        st_sums = st_sums / (len(timestamps))
        st_sums = st_sums / (st_sums.max() + 1e-8)

        st_sums = torch.log(100*st_sums + 1) / 2

        print("ST sums:", st_sums.shape,st_sums.mean().item(), st_sums.min().item(), st_sums.max().item())
        print("Num invisibles:",invisibles.sum())
        bad_points = (st_sums < 0.001) & ~invisibles
        print("Num < 0.001",bad_points.sum().item())

        clone_split_prune(gaussians,prunes = invisibles|bad_points)


    def add_densification_stats_grad(self,*,gaussians,iteration,viewspace_point_grad,update_filter, radii,avg_t_gradient):
        pass

    def get_save_vars(self,gaussians):
        return []

    def needs_densification_or_pruning(self, gaussians, iteration):
        iterations = self._get_option("iterations",-1)

        if type(iterations)==int:
            iterations=[iterations]
        if iteration in iterations:# or not hasattr(self,"done"):
            #self.done = True
            return {"densify": False, "prune": True, "final_prune": False}
        else:
            return None



                         



