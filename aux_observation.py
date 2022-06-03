#!/usr/bin/env python3

import numpy as np
import astropy.units as u

import pandas as pd
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import AltAz, SkyCoord, EarthLocation

class observation :
    """
    A class used to create final simulation data 

    :param sim_object : simulation object 
    :type : simulation_object
    :param src_object: source data
    :type: source_object

    """

    def __init__(self, sim_object, src_object, time_start='2022-01-16T22:00:00', 
    time_stop='2022-01-16T22:20:00', RA = 84.0331, Dec = +22.0145 ): 

        self.sim_object = sim_object
        self.src_object = src_object

        self.time_start = Time(time_start, format='isot', scale='utc')
        self.time_stop  = Time(time_stop, format='isot', scale='utc')
        self.tobs = (self.time_stop - self.time_start).to(u.s) 

        self.RA = RA
        self.Dec = Dec

    def center_camera(self) :
        src_coord = SkyCoord(self.src_object.RA, self.src_object.Dec, unit="deg")
        pointing_coord = SkyCoord(self.RA, self.Dec, unit="deg")
        observation_localisation = EarthLocation(
        lat=28.761758*u.deg,
        lon=-17.890659*u.deg,
        height=2200*u.m
        )
        observation_time = Time(self.time_start)
        aa = AltAz(location=observation_localisation, obstime= observation_time)
        src_coord_AA = src_coord.transform_to(aa)
        pointing_coord_AA = pointing_coord.transform_to(aa)
        sep = pointing_coord_AA.spherical_offsets_to(src_coord_AA)
        return sep[0].deg, sep[1].deg

    def area_scale (self) : 
        """ Calculate area scale (simulation data area / field of view area)
        
        :returns: area scale
        :rtype: float
        """
        area_fov = self.src_object.src_area()
        area_sim = self.sim_object.sim_area
        return area_sim / area_fov


    def spatial_weights (self, cam_x, cam_y) : 
        """ Calculate spatial weights

        :param cam_x : coordinate x of the camera
        :type: list of float
        :param cam_y : coordinate y of the camera
        :type: list of float
        :returns: weights
        :rtype: list 
        """
        x0 = self.center_camera(self)[0] *u.deg
        y0 = self.center_camera(self)[1] *u.deg
        rmax = self.src_object.rmax
        rmin = self.src_object.rmin
        r = np.sqrt((cam_x.to(u.deg) - x0)**2 + (cam_y.to(u.deg) - y0)**2)
        if self.src_object.shape == "disk" : 
            return (r < rmax)
        if self.src_object.shape == "gauss" : 
            return np.exp(- r.to(u.deg)**2 / rmax.to(u.deg)**2)
        else :
            return (r > rmin ) & (r < rmax)


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
        weights_total = self.spatial_weights(self, self.sim_object.cam_x, self.sim_object.cam_y) * spectral_weights * self.area_scale(self) 
        return weights_total

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
        weights_tot = self.background_weighting(self)
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
        weights_tot = self.weighting() # self.weighting(self)[1] * self.weighting(self)[0] * self.weighting(self)[2]
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

        return sim_data, index

    def total(self, write=False, filename='dl2_LST-1.Run99999.h5') :
        """ Return final simulation data (data+background) with all informations

        :param write : save simulation data in hdf file, if write is True
        :type write: bool, optional
        :param filename : the name of the simulation data file saved, if 'dl2_LST-1.Run99999.h5' uses default
        :type filename: string, optional
        :returns: final data table 
        :rtype: DataFrame
        """
        
        df_background = self.final_background_sim(self)
        df_data = self.final_sim(self) 
        total = pd.concat([df_data, df_background])
        print('etape 1')
        total = total.sort_values(by = 'dragon_time')
        print('etape 2')
        size= len(total)
        print('etape 3 size =', size)

        pointing_coord = SkyCoord(self.RA, self.Dec, unit="deg")
        observation_localisation = EarthLocation(
            lat=28.761758*u.deg,
            lon=-17.890659*u.deg,
            height=2200*u.m
            )
        dt = total['dragon_time'].to_numpy()
        observation_time = [ datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dt]
        print('etape 4')
        aa = AltAz(location=observation_localisation, obstime= observation_time)
        pointing_coord_AA = pointing_coord.transform_to(aa)
        print('etape 5')
        az  = (pointing_coord_AA.az).to('rad').value
        alt = (pointing_coord_AA.alt).to('rad').value
        print('etape 6')
        total = total.drop(columns=["az_tel", "alt_tel"])
        print('etape 7')
        total = total.assign(az_tel=az, alt_tel=alt)
        print('etape 8')
        

        if write==True:
            total.to_hdf(filename, 'dl2/event/telescope/parameters/LST_LSTCam')
        else:
            pass

        return total




