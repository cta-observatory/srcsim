#!/usr/bin/env python3

import numpy as np
import astropy.units as u

class observation :

    def __init__(self, sim_object, src_object): 

        self.sim_object = sim_object
        self.src_object = src_object


    def weighting(self) : 
        """ Return the final weight for the histogram of energy of MC data

        :param sim_object: simulation data
        :type: simulation_object
        :param src_object: source data
        :type: source_object
        :returns: final weights
        :rtype: list 
        """
        spectral_weights = self.src_object.tobs * self.src_object.spectrum(self.sim_object.mc_energy) / self.sim_object.powerlaw_MC_data() 
        return self.src_object.spatial_weights(self.sim_object.cam_x, self.sim_object.cam_y) * spectral_weights * self.src_object.area_scale()

    def final_sim(self) :

        mc_energy = self.sim_object.mc_energy
        weights_tot = observation.weighting(self)
        probability = weights_tot / weights_tot.sum()
        n_expected_events = int(weights_tot.sum())
        index = np.random.choice(
                    np.arange(len(mc_energy)),
                    size=n_expected_events,
                    p=probability
                )
        return self.sim_object.data.iloc[index]



