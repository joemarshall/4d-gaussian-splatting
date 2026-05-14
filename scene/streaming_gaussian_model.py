from .gaussian_model import GaussianModel

# needs to:
# - keep all parameters, gradients etc. in memory mapped file based backing stores.
#   load gaussians for a time range to cuda
#   copy back from cuda -> memory mapped file 
#   move gaussians for a time range to a different time range if the mean shifts out

# n.b. maybe make: streamingtensor - tensor backed by memmapped file and ordered by time

# call load_gaussians only after zeroing gradients, so that we don't have to save gradients

class Streaming4DGaussianModel(GaussianModel):

    def _find_tensors(self):
        self.all_tensors = []
        def all_args_fn()

        self._make_save_or_restore_calls(restore_fn, model_args)


        for name, param in self.parame
            if "gaussians" in name:
                self.all_tensors.append(param)
        assert len(self.all_tensors) == 1, "Expected exactly one tensor containing gaussians"
        self.gaussians = self.all_tensors[0]

    def load_gaussians_for_time_range(self,time_range):
        if not hasattr(self, "all_tensors"):
            self.
            self.gaussians.cpu()
        for x in all_gaussians:
