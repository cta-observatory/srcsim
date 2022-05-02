#!/usr/bin/env python3

import numpy as np
import astropy.units as u

powerlaw_spectrum_parameters = {"type" : "powerlaw", "E_0" :  1 * u.TeV, 
    "norm" : 3.80e-11 * 1/(u.TeV * u.cm**2 * u.s), "index" : -2.21, "curve" : 0, "E_cut" : 0 }

cutoff_spectrum_parameters = {"type" : "cutoff", "E_0" :  1 * u.TeV, 
    "norm" : 3.80e-11 * 1/(u.TeV * u.cm**2 * u.s), "index" : -2.21, "curve" : 0, "E_cut" : 6.0 * u.TeV}

def weighting(sim_object, src_object) : 
        
        spectral_weights = src_object.tobs * src_object.spectrum(sim_object.mc_energy) / sim_object.powerlaw_MC_data() 

        return src_object.spatial_weights(sim_object.cam_x, sim_object.cam_y) * spectral_weights * src_object.area_scale()


