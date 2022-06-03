#!/usr/bin/env python3

from aux_observation import *
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

    def __init__(self, filename_gamma, filename_proton):

        self.data = pd.read_hdf(filename_gamma, key=dl2_params_lstcam_key)
        self.data_config = pd.read_hdf(filename_gamma, "simulation/run_config")
        self.data_proton = pd.read_hdf(filename_proton, key=dl2_params_lstcam_key)
        self.data_config_proton = pd.read_hdf(filename_proton, "simulation/run_config")

        self.powerlaw_parameter_gamma = { "type" : "gamma", "E_min" : self.data_config['energy_range_min'][0] * u.TeV , "E_max" : 
        self.data_config['energy_range_max'][0] * u.TeV, "E_0" :  np.sqrt(self.data_config['energy_range_max'][0] * u.TeV * 
        self.data_config['energy_range_min'][0] * u.TeV) , "N" : self.data_config['num_showers'][0] * self.data_config['shower_reuse'][0] 
        * len(self.data_config['shower_reuse']), "index" : self.data_config['spectral_index'][0], "radius" : self.data_config['max_scatter_range'][0] * u.m  }

        self.powerlaw_parameter_proton = { "type" : "proton", "E_min" : self.data_config_proton['energy_range_min'][0] * u.TeV , "E_max" : 
        self.data_config_proton['energy_range_max'][0] * u.TeV, "E_0" :  np.sqrt(self.data_config_proton['energy_range_max'][0] * u.TeV * 
        self.data_config_proton['energy_range_min'][0] * u.TeV) , "N" : self.data_config_proton['num_showers'][0] * self.data_config_proton['shower_reuse'][0] 
        * len(self.data_config_proton['shower_reuse']), "index" : self.data_config_proton['spectral_index'][0],
        "radius" : self.data_config_proton['max_scatter_range'][0] * u.m  }

        self.f = 28 * u.m  
        self.cam_x = (self.data['reco_src_x'].to_numpy() * u.m / self.f) * u.rad           
        self.cam_y = (self.data['reco_src_y'].to_numpy() * u.m / self.f) * u.rad
        self.viewcone = self.data_config['max_viewcone_radius'][0] * u.deg

        self.mc_energy = self.data['mc_energy'].to_numpy() * u.TeV
        self.mc_energy_proton = self.data_proton['mc_energy'].to_numpy() * u.TeV

        self.sim_area = 2 * np.pi * (1 - np.cos(self.viewcone))

       

    def powerlaw_MC_data(self, type = "gamma") :

        """Return powerlaw of MC damma data 

        :returns: power law normed
        :rtype: list of float
        """
        if type == "proton" : 
            parameter=self.powerlaw_parameter_proton
            S = np.pi * (parameter["radius"].to(u.cm))**2 
            integral = ( 1/(parameter["index"]+1) ) * parameter["E_0"].to(u.TeV) * (( parameter["E_max"] / parameter["E_0"] )**(parameter["index"]+1) - (parameter["E_min"]/parameter["E_0"])**(parameter["index"]+1) ) 
            norm = parameter["N"]/(S*integral) 
            return norm * (self.mc_energy_proton/parameter["E_0"]).decompose()**parameter["index"]
        else :
            parameter=self.powerlaw_parameter_gamma
            S = np.pi * (parameter["radius"].to(u.cm))**2 
            integral = ( 1/(parameter["index"]+1) ) * parameter["E_0"].to(u.TeV) * (( parameter["E_max"] / parameter["E_0"] )**(parameter["index"]+1) - (parameter["E_min"]/parameter["E_0"])**(parameter["index"]+1) ) 
            norm = parameter["N"]/(S*integral) 
            return norm * (self.mc_energy/parameter["E_0"]).decompose()**parameter["index"]
        




    



