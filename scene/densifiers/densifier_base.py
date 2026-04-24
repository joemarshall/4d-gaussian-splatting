class DensifierBase:
    def __init__(self, opt, name):
        self.options = opt
        self.name = name

    def training_setup(self, gaussians, reset_accumulated_gradients):
        """ Any setup that needs to be done before training starts can be done here, such as initializing accumulators."""
        pass

    def densify_and_prune(self, iteration, scene, gaussians, radii,pipe, bg,*,options):
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

    def _get_option(self, option_name, default_value):
        """Helper method to get an option value with a default and a potential
        per_densifier override with prefix name_."""
        # n.b. can't just getattr because if it isn't set then
        # we want to return default value from base class rather than
        # always returning default value from our customized option
        retval = getattr(self.options,self.name+"_"+option_name,None)
        if retval is None: 
            retval = getattr(self.options, option_name,None)
            if retval is None:
                return default_value
        #print("Options for {}: {}={}".format(self.name, option_name, retval))
        return retval

    def needs_densification_or_pruning(self, gaussians, iteration):
        """ Determine if densification needs to be called based on the current iteration and densifier options. """
        densification_interval = self._get_option("densification_interval", 100)
        densify_from_iter = self._get_option("densify_from_iter", 0)
        densify_until_iter = self._get_option("densify_until_iter", 1e9)

        # print(iteration, densify_from_iter, densify_until_iter, densification_interval)

        needs_densify = iteration >= densify_from_iter and iteration < densify_until_iter and iteration % densification_interval == 0
        needs_prune = needs_densify
        final_prune = self._get_option("final_prune_from_iter", -1) > 0 and iteration >= self._get_option("final_prune_from_iter", -1) and iteration % self._get_option("final_prune_interval", 1000) == 0

        if self._get_option("densify_until_num_points", 0) > 0 and  gaussians.get_xyz.shape[0] >= self._get_option("densify_until_num_points", 0):
            needs_densify = False

        if needs_densify or needs_prune or final_prune:
            return {"densify":needs_densify, "prune":needs_prune, "final_prune":final_prune}
        else:
            return None


    def densification_postfix(self, gaussians):
        """ resize accumulation variables after size change"""
        if len(self.get_save_vars(gaussians))>0:
            print("Called base densification_postfix, but this should be implemented by subclasses that have accumulation variables to resize after densification.")

    def prune_points(self, gaussians,valid_points_mask):
        """ prune points according to the mask"""
        if len(self.get_save_vars(gaussians))>0:
            print("Called base prune_points, but this should be implemented by subclasses that have accumulation variables to resize after pruning.")

    def per_iteration(self, iteration, scene, gaussians, radii, pipe, bg):
        """ Any per-iteration code that needs to be run for this densifier can be put here. 
        This is called every iteration, even after densification has finished. Useful for cleanup (see fastgs final pruning)."""
        pass

    def apply_debug_colour(self, gaussians, scene,pipe,bg, debug_type=""):
        """ change gaussian colour for some kind of debugging purpose - debug_type is a string that can be used to specify what kind of debugging info to show
        """
        return None
