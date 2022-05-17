#!/usr/bin/env python3

import numpy as np
import astropy.units as u
import time
import pandas as pd
from astropy.time import Time

class observation :
    """
    A class used to create final simulation data 

    :param sim_object : simulation object 
    :type : simulation_object
    :param src_object: source data
    :type: source_object

    """

    def __init__(self, sim_object, src_object): 

        self.sim_object = sim_object
        self.src_object = src_object


    def weighting(self) : 
        """ Return the final weight for the histogram of energy of MC data

        :returns: final weights
        :rtype: list 
        """
        spectral_weights = self.src_object.tobs * self.src_object.spectrum(self.sim_object.mc_energy) / self.sim_object.powerlaw_MC_data() 
        return self.src_object.spatial_weights(self.sim_object.cam_x, self.sim_object.cam_y) * spectral_weights * self.src_object.area_scale()

    def final_sim(self, write=False, filename='simulation.h5') :
        """ Return final simulation data with all informations

        :param write : save simulation data in hdf file, if write is True
        :type write: bool, optional
        :param filename : the name of the simulation data file saved, if 'simulation.h5' uses default
        :type filename: string, optional
        :returns: final simulation data table 
        :rtype: DataFrame
        """

        #step: random weights
        mc_energy = self.sim_object.mc_energy
        weights_tot = observation.weighting(self)
        probability = weights_tot / weights_tot.sum()
        n_expected_events = int(weights_tot.sum())
        index = np.random.choice(
                    np.arange(len(mc_energy)),
                    size=n_expected_events,
                    p=probability
                )
        
        sim_data = self.sim_object.data.iloc[index]

        #step: add dragon_time and delta_t columns 
        lamb= 5500
        size= len(sim_data)
        start_time = Time("2022-05-16 01:00:00")
        stop_time = Time("2022-05-16 02:00:00")
        delta_t = np.random.exponential(1/lamb,size)
        dragon_time = np.linspace(start_time.to_value('mjd'),stop_time.to_value('mjd'), size)
        sim_data = sim_data.assign(delta_t=delta_t, dragon_time=dragon_time)

        #step: writing or not of the file
        if write==True:
            sim_data.to_hdf(filename, 'dl2/event/telescope/parameters/LST_LSTCam')
        else:
            pass

        return sim_data



