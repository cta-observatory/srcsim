#!/usr/bin/env python3

from aux import *
from lstchain.io.io import dl2_params_lstcam_key
import pandas as pd
import numpy as np
import astropy.units as u


class simulation_object :

    def __init__(self, filename):
        data_config = pd.read_hdf(filename, "simulation/run_config")
        data = pd.read_hdf(filename, key=dl2_params_lstcam_key)

        self.E_min = data_config['energy_range_min'][0] * u.TeV
        self.E_max = data_config['energy_range_max'][0] * u.TeV
        self.radius = data_config['max_scatter_range'][0] * u.m 
        self.mc_energy = data['mc_energy'].to_numpy() * u.TeV
        self.index = data_config['spectral_index'][0]
        self.N = data_config['num_showers'][0] * data_config['shower_reuse'][0] * len(data_config['shower_reuse'])
        self.E_0 = np.sqrt(self.E_max * self.E_min) 
        self.S = np.pi * (self.radius.to(u.cm))**2 
        self.f = 27 * u.m  
        self.cam_x = (data['reco_src_x'].to_numpy() * u.m / self.f) * u.rad           
        self.cam_y = (data['reco_src_y'].to_numpy() * u.m / self.f) * u.rad
       

    def powerlaw_MC_data (self) :
        integral = ( 1/(self.index+1) ) * self.E_0.to(u.TeV) * ( (self.E_max/self.E_0)**(self.index+1) - (self.E_min/self.E_0)**(self.index+1) ) 
        norm = self.N/(self.S*integral) 
        return norm * (self.mc_energy/self.E_0).decompose()**self.index

    


    



