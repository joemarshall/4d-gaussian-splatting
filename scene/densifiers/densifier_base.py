class DensifierBase:
    def __init__(self, opt):
        self.options = opt

    def training_setup(self, gaussians, reset_accumulated_gradients):
        """ Any setup that needs to be done before training starts can be done here, such as initializing accumulators."""
        pass

    def densify_and_prune(self, iteration, scene, gaussians, radii,pipe, bg):
        """ Do densification and pruning on gaussians using this method. """
        raise NotImplementedError("Densify method must be implemented by subclasses.")
    
    def add_densification_stats(self,*,gaussians,iteration,viewspace_point_tensor,visibility_filter,radii,avg_t_gradient):
        """Collect any necessary statistics from each iteration here for next densification call.
           n.b. by default this calls through to add_densification_stats_grad """
        viewspace_point_grad = viewspace_point_tensor.grad[:,:2].unsqueeze(1)
        self.add_densification_stats_grad(gaussians = gaussians, iteration = iteration, viewspace_point_grad = viewspace_point_grad, 
                                          update_filter = visibility_filter, radii = radii, avg_t_gradient = avg_t_gradient)

    def add_densification_stats_grad(self,*,gaussians,iteration,viewspace_point_grad,update_filter, radii,avg_t_gradient):
        """Collect any necessary statistics from each iteration here for next densification call."""
        print("Warning: add_densification_stats_grad not implemented for this densifier, so no stats will be collected for next densification step.")

    def get_save_vars(self,gaussians):
        """Return variables that should be saved with the scene, as a list of attribute names."""
        raise NotImplementedError("Get save vars method must be implemented by subclasses.")
    
    def densification_postfix(self, gaussians):
        """ resize accumulation variables after size change"""
        print("Called base densification_postfix, but this should be implemented by subclasses that have accumulation variables to resize after densification.")

    def prune_points(self, valid_points_mask):
        """ prune points according to the mask"""
        print("Called base prune_points, but this should be implemented by subclasses that have accumulation variables to resize after pruning.")
        
    def per_iteration(self, iteration, scene, gaussians, radii, pipe, bg):
        """ Any per-iteration code that needs to be run for this densifier can be put here. 
        This is called every iteration, even after densification has finished. Useful for cleanup (see fastgs final pruning)."""
        pass

    def apply_debug_colour(self, gaussians, scene,pipe,bg, debug_type=""):
        """ change gaussian colour for some kind of debugging purpose - debug_type is a string that can be used to specify what kind of debugging info to show
        """
        return None
