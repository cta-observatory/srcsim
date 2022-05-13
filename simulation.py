#!/usr/bin/env python3

from aux import *
from lstchain.io.io import dl2_params_lstcam_key
import pandas as pd
import numpy as np
import astropy.units as u


class simulation_object :
    """
    A class used to create simulated object issus to MC data 

    :param filename: name of MC data file
    :type filename: str
    :ivar E_min : the energy range mininum (in TeV)
    :vartype: int
    :ivar E_max : the energy range maximum (in TeV)
    :vartype: int
    :ivar radius : the radius of the effective area (in m)
    :vartype: float
    :ivar mc_energy : the energy of the mc data (in TeV)
    :vartype: list of float
    :ivar index : the spectral index 
    :vartype: float
    :ivar N : the number of showers
    :vartype: int
    :ivar E_0 : (in TeV)
    :vartype: float
    :ivar S : the surface of the effective area (in cm2)
    :vartype: float
    :ivar f : the focal lenght of the telescope 
    :vartype: int
    :ivar cam_x : coordinate x of the camera
    :vartype: list of float
    :ivar cam_y : coordinate y of the camera
    :vartype: list of float

    """

    def __init__(self, filename):
        self.data_config = pd.read_hdf(filename, "simulation/run_config")
        self.data = pd.read_hdf(filename, key=dl2_params_lstcam_key)

        self.E_min = self.data_config['energy_range_min'][0] * u.TeV
        self.E_max = self.data_config['energy_range_max'][0] * u.TeV
        self.radius = self.data_config['max_scatter_range'][0] * u.m 
        self.mc_energy = self.data['mc_energy'].to_numpy() * u.TeV
        self.index = self.data_config['spectral_index'][0]
        self.N = self.data_config['num_showers'][0] * self.data_config['shower_reuse'][0] * len(self.data_config['shower_reuse'])
        self.E_0 = np.sqrt(self.E_max * self.E_min) 
        self.S = np.pi * (self.radius.to(u.cm))**2 
        self.f = 27 * u.m  
        self.cam_x = (self.data['reco_src_x'].to_numpy() * u.m / self.f) * u.rad           
        self.cam_y = (self.data['reco_src_y'].to_numpy() * u.m / self.f) * u.rad
       

    def powerlaw_MC_data (self) :
        """Return powerlaw of MC data 

        :returns: power law normed
        :rtype: list of float
        """
        integral = ( 1/(self.index+1) ) * self.E_0.to(u.TeV) * ( (self.E_max/self.E_0)**(self.index+1) - (self.E_min/self.E_0)**(self.index+1) ) 
        norm = self.N/(self.S*integral) 
        return norm * (self.mc_energy/self.E_0).decompose()**self.index

    


    



