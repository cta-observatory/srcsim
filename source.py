#!/usr/bin/env python3

import numpy as np
import astropy.units as u
from simulation import *
from aux import *

class source_object :

    def __init__(self, tobs=20*u.min, spectrum_parameters = {"type" : "log_parabola", "E_0" :  1 * u.TeV, 
    "norm" : 3.23e-11 * 1/(u.TeV * u.cm**2 * u.s), "index" : - 2.47, "curve" : - 0.24, "E_cut" : 0 }, 
    rmax=0.7, rmin=0.1, x0=0, y0=0, offset=0.4 *u.deg) :
        self.x0 = x0 *u.deg
        self.y0 = y0 *u.deg
        self.offset = offset * u.deg
        self.tobs = tobs.to("s")
        self.rmax = rmax * u.deg
        self.rmin = rmin * u.deg
        self.spectrum_parameters = spectrum_parameters

    
    def spatial_weights (self, cam_x, cam_y, shape="disk" ) : 
        r = np.sqrt((cam_x.to(u.deg) - self.x0)**2 + (cam_y.to(u.deg) - self.y0)**2)
        if shape == "gaussian" : 
            return np.exp(-r**2 / (self.rmax* u.deg)**2)
        else :
            return (r > self.rmin ) & (r < self.rmax)

    def area_scale (self) : 
        area_fov = 2 * np.pi * (np.cos(self.rmin) - np.cos(self.rmax))
        area_sim = 2 * np.pi * (1 - np.cos(6*u.deg))
        return area_sim / area_fov

    def spectrum (self, energy) :
        if self.spectrum_parameters["type"] == "cutoff" : 
            return self.spectrum_parameters["norm"] * (energy/self.spectrum_parameters["E_0"]).decompose()**self.spectrum_parameters["index"] * np.exp((-energy/self.spectrum_parameters["E_cut"]).decompose())
        
        if self.spectrum_parameters["type"] == "powerlaw" :
            return self.spectrum_parameters["norm"] * (energy/self.spectrum_parameters["E_0"]).decompose()**self.spectrum_parameters["index"]
        else : 
            return self.spectrum_parameters["norm"] * (energy/self.spectrum_parameters["E_0"]).decompose()**(self.spectrum_parameters["index"] + self.spectrum_parameters["curve"]*(np.log((energy/self.spectrum_parameters["E_0"]).decompose())))