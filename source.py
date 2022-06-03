#!/usr/bin/env python3

import numpy as np
import astropy.units as u
from simulation import *
from aux_observation import *

class source_object :

    """
    A class used to contain parameters of the source object 

    :param tobs : time observation of source data (in second), if 20*u.min uses default
    :type tobs: float, optional
    :param spectrum_parameters : parameters of the source spectrum (put astropy units), if log parabola function uses default
    :type spectrum_parameters: dictionnary, optional
    :param rmax : radius max for spatial weights (in deg), if 0.7 uses default
    :type rmax: float, optional
    :param rmin : radius min for spatial weights (in deg), if 0.1 uses default
    :type rmin: float, optional 
    :param x0 : coordinate x of the center of the camera, if 0 uses default
    :type x0: float, optional 
    :param y0 : coordinate y of the center of the camera, if 0 uses default
    :type y0: float, optional  
    :param offset : offset of source data, if 0.4 uses default
    :type offset: float, optional  
    :param shape : shape of selected data, if "ring" uses default
    :type shape: str, optional   

    """

    def __init__(self, spectrum_parameters = {"type" : "log_parabola", "E_0" :  1 * u.TeV, 
    "norm" : 3.23e-11 * 1/(u.TeV * u.cm**2 * u.s), "index" : - 2.47, "curve" : - 0.24, "E_cut" : 0 }
    , offset=0.4, shape = "ring", rmax = 0.7* u.deg, rmin = 0.1 *u.deg ) :
        self.RA = 83.6331 
        self.Dec = +22.0145 
        self.offset = offset * u.deg

        self.spectrum_parameters = spectrum_parameters

        self.powerlaw_spectrum_parameters = {"type" : "powerlaw", "E_0" :  1 * u.TeV, 
        "norm" : 3.80e-11 * 1/(u.TeV * u.cm**2 * u.s), "index" : -2.21, "curve" : 0, "E_cut" : 0 }
        self.cutoff_spectrum_parameters = {"type" : "cutoff", "E_0" :  1 * u.TeV, 
        "norm" : 3.80e-11 * 1/(u.TeV * u.cm**2 * u.s), "index" : -2.21, "curve" : 0, "E_cut" : 6.0 * u.TeV}

        self.rmin = rmin
        self.rmax = rmax
        self.shape = shape
    
    def src_area (self) : 
        if self.shape == "disk" : 
            area_fov = 2 * np.pi * (1 - np.cos(self.rmax))
        if self.shape== "gauss" : 
            area_fov = 2 * np.pi * (1 - np.cos(1.5 * self.rmax))
        else :
            area_fov = 2 * np.pi * (np.cos(self.rmin) - np.cos(self.rmax))
        return area_fov

    
    def spectrum (self, energy) :
        """ Calculate the differential photon spectrum of the source
        
        :param energy: energy of MC data 
        :type: list of float 
        :returns: differential photon spectrum normed
        :rtype: float
        """
        if self.spectrum_parameters["type"] == "cutoff" : 
            return self.spectrum_parameters["norm"] * (energy/self.spectrum_parameters["E_0"]).decompose()**self.spectrum_parameters["index"] * np.exp((-energy/self.spectrum_parameters["E_cut"]).decompose())
        
        if self.spectrum_parameters["type"] == "powerlaw" :
            return self.spectrum_parameters["norm"] * (energy/self.spectrum_parameters["E_0"]).decompose()**self.spectrum_parameters["index"]
        else : 
            return self.spectrum_parameters["norm"] * (energy/self.spectrum_parameters["E_0"]).decompose()**(self.spectrum_parameters["index"] + self.spectrum_parameters["curve"]*(np.log((energy/self.spectrum_parameters["E_0"]).decompose())))