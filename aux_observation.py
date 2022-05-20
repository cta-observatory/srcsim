#!/usr/bin/env python3

import numpy as np
import astropy.units as u

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

    def __init__(self, sim_object, src_object, time_start='2022-05-16T01:00:00', time_stop='2022-05-16T01:20:00'): 

        self.sim_object = sim_object
        self.src_object = src_object
        self.time_start = Time(time_start, format='isot', scale='utc')
        self.time_stop  = Time(time_stop, format='isot', scale='utc')
        self.tobs = (self.time_stop - self.time_start).to(u.s) 

    def background_weighting(self) :
        """ Return the final weight of background 

        :returns: final weights
        :rtype: list 
        """
        F_0 = (7.58e-5 / (u.GeV * u.m**2 * u.s * u.sr)).to(u.TeV**-1 * u.cm**-2 * u.s**-1 * u.sr**-1)
        gamma = 2.772
        E_b = 0.48 * u.TeV
        delta_gamma = 0.173
        smooth = 5
        energy = self.sim_object.mc_energy_proton
        fov = 2 * np.pi * (1 - np.cos(6*u.deg)) * u.sr

        proton_function = F_0 * (energy/u.TeV)**(-gamma) * (1+(energy/E_b)**smooth)**(delta_gamma/smooth) * fov

        return self.tobs * proton_function / self.sim_object.powerlaw_MC_data(type="proton") 

    def weighting(self) : 
        """ Return the final weight of simulation data

        :returns: final weights
        :rtype: list 
        """
        spectral_weights = self.tobs * self.src_object.spectrum(self.sim_object.mc_energy) / self.sim_object.powerlaw_MC_data() 
        return self.src_object.spatial_weights(self.sim_object.cam_x, self.sim_object.cam_y) * spectral_weights * self.src_object.area_scale() 

    def final_background_sim(self, write=False, filename='background.h5') : 
        """ Return final background data with all informations

        :param write : save simulation data in hdf file, if write is True
        :type write: bool, optional
        :param filename : the name of the simulation data file saved, if 'background.h5' uses default
        :type filename: string, optional
        :returns: final background data table 
        :rtype: DataFrame
        """

        #step: random weights
        mc_energy = self.sim_object.mc_energy_proton
        weights_tot = observation.background_weighting(self)
        probability = weights_tot / weights_tot.sum()
        n_expected_events = int(weights_tot.sum())
        index = np.random.choice(
                    np.arange(len(mc_energy)),
                    size=n_expected_events,
                    p=probability
                )
        
        background_data = self.sim_object.data_proton.iloc[index]

        #step: add dragon_time and delta_t columns 
        lamb= 12e3
        size= len(background_data)

        delta_t = np.random.exponential(1/lamb,size)
        dragon_time = np.linspace(self.time_start.unix, self.time_stop.unix, size) 
        background_data = background_data.assign(delta_t=delta_t, dragon_time=dragon_time)

        #step: writing or not of the file
        if write==True:
            background_data.to_hdf(filename, 'dl2/event/telescope/parameters/LST_LSTCam')
        else:
            pass

        return background_data




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
        lamb= 12e3
        size= len(sim_data)

        delta_t = np.random.exponential(1/lamb,size)
        dragon_time = np.linspace(self.time_start.unix, self.time_stop.unix, size)
        sim_data = sim_data.assign(delta_t=delta_t, dragon_time=dragon_time)

        #step: writing or not of the file
        if write==True:
            sim_data.to_hdf(filename, 'dl2/event/telescope/parameters/LST_LSTCam')
        else:
            pass

        return sim_data

    def total(self, write=False, filename='dl2_LST-1.Run99999.h5') :
        """ Return final simulation data (data+background) with all informations

        :param write : save simulation data in hdf file, if write is True
        :type write: bool, optional
        :param filename : the name of the simulation data file saved, if 'dl2_LST-1.Run99999.h5' uses default
        :type filename: string, optional
        :returns: final data table 
        :rtype: DataFrame
        """

        df_background = observation.final_background_sim(self)
        df_data = observation.final_sim(self) 
        total = pd.concat([df_data, df_background])
        total = total.sort_values(by = 'dragon_time')

        if write==True:
            total.to_hdf(filename, 'dl2/event/telescope/parameters/LST_LSTCam')
        else:
            pass

        return total




