import glob
import yaml
import functools
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation

from ..run import SkyDataRun
from .helpers import get_trajectory, get_time_intervals, read_obs, enforce_max_interval_length


class AltAzBoxGenerator:
    """
    Generator of the sky runs within a given alt/az box
    """

    @classmethod
    def get_runs_from_config(cls, config):
        """
        Returns runs corresponding to the specified config.

        Parameters
        ----------
        config: dict
            Configuration dictionary

        Returns
        -------
        runs: tuple
            Generated runs
        """

        if isinstance(config, str):
            cfg = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        else:
            cfg = config
        
        tel_pos = SkyCoord(
            u.Quantity(cfg['pointing']['ra']),
            u.Quantity(cfg['pointing']['dec']),
            frame='icrs'
        )

        if cfg['pointing']['wobble'] is not None:
            wobble_offset = u.Quantity(cfg['pointing']['wobble']['offset'])
            wobble_start_angle = u.Quantity(cfg['pointing']['wobble']['start_angle'])
            wobble_count = cfg['pointing']['wobble']['count']
        else:
            wobble_offset = None
            wobble_start_angle = None
            wobble_count = None

        azmin = u.Quantity(cfg['box']['az']['min'])
        azmax = u.Quantity(cfg['box']['az']['max'])
        altmin = u.Quantity(cfg['box']['alt']['min'])
        altmax = u.Quantity(cfg['box']['alt']['max'])

        tstart = Time(cfg['time']['start'])
        duration = u.Quantity(cfg['time']['duration'])
        accuracy = u.Quantity(cfg['time']['accuracy'])
        max_run_duration = u.Quantity(cfg['time']['max_run_duration']) if cfg['time']['max_run_duration'] is not None else None
        obsloc = EarthLocation(
            lat=u.Quantity(cfg['location']['lat']),
            lon=u.Quantity(cfg['location']['lon']),
            height=u.Quantity(cfg['location']['height']),
        )

        return cls.get_runs(obsloc, tel_pos, duration, altmin, altmax, azmin, azmax, tstart, accuracy, max_run_duration, wobble_offset, wobble_start_angle, wobble_count)

    @classmethod
    def get_runs(cls, obsloc, tel_pos, tobs, altmin, altmax, azmin, azmax, tstart=None, accuracy=1*u.minute, max_run_duration=None, wobble_offset=None, wobble_start_angle=None, wobble_count=None):
        """
        Generates runs corresponding to specified sky constraints.

        Parameters
        ----------
        obsloc: EarthLocation
            telescope location
        tel_pos: SkyCoord
            equatorial telscope pointing coordinates
        tobs: u.Quantity
            total observation duration
        altmin: u.Quantity
            minimal telescope trajectory altitude to use.
        altmax: u.Quantity
            maximal telescope trajectory altitude to use.
        azmin: u.Quantity
            minimal telescope trajectory azimuth to use.
        azmax: u.Quantity
            maximal telescope trajectory azimuth to use.
        tstart: Time
            start time for generated runs
        accuracy: u.Quantity
            trajectory time step to use in calculations
        max_run_duration: u.Quantity
            maximal length of a single run to allow
        wobble_offset: u.Quantity
            wobble offset to apply
        wobble_start_angle: u.Quantity
            positional angle of the first wobble
        wobble_count: int
            number of wobbles to assume. If none, no wobbles 
            will be generated and runs will center at tel_pos

        Returns
        -------
        runs: tuple
            Generated runs
        """

        wobble_params = (wobble_offset, wobble_start_angle, wobble_count)
        n_params_set = sum([par is not None for par in wobble_params])
        use_wobble = n_params_set == len(wobble_params)

        if not use_wobble and n_params_set > 0:
            raise ValueError(f"must specify all or none of the ['wobble_offset', 'wobble_start_angle', 'wobble_count'] parameters")

        if tstart is None:
            tstart = Time('1970-01-01')

        tel_altaz = get_trajectory(
            tel_pos,
            tstart,
            tstop=tstart + 1 * u.d,
            time_step=accuracy,
            obsloc=obsloc
        )
        
        # Second pass with the start on the source anti-culmination
        tstart = tel_altaz.obstime[tel_altaz.alt.argmin()]
        tel_altaz = get_trajectory(
            tel_pos,
            tstart,
            tstop=tstart + 1 * u.d,
            time_step=accuracy,
            obsloc=obsloc
        )
        
        tstarts, tstops = get_time_intervals(tel_altaz, altmin, altmax, azmin, azmax, accuracy)
        run_durations = tstops - tstarts
        
        nsequences = (tobs / np.sum(run_durations)).decompose()
        ndays_full = np.floor(nsequences)
        ndays_total = np.ceil(nsequences)
        
        if ndays_full > 0:
            # Third pass - to cover the simulation time interval with fully completed runs
            tel_altaz = get_trajectory(
                tel_pos,
                tstart,
                tstop=tstart + ndays_full * u.d,
                time_step=accuracy,
                obsloc=obsloc
            )
            tstarts, tstops = get_time_intervals(tel_altaz, altmin, altmax, azmin, azmax, accuracy)
            remaining_tobs = tobs - np.sum(tstops - tstarts)
        
            # Fourth pass - additional incomplete runs
            _tel_altaz = get_trajectory(
                tel_pos,
                tstart=tstart + ndays_full * u.d,
                tstop=tstart + ndays_total * u.d,
                time_step=accuracy,
                obsloc=obsloc
            )
            _tstarts, _tstops = get_time_intervals(_tel_altaz, altmin, altmax, azmin, azmax, accuracy)
            _tstops = Time(_tstarts + remaining_tobs / (len(_tstarts)))

            tstarts = Time([tstarts, _tstarts])
            tstops = Time([tstops, _tstops])

        else:
            tel_altaz = get_trajectory(
                tel_pos,
                tstart,
                tstop=tstart + ndays_total * u.d,
                time_step=accuracy,
                obsloc=obsloc
            )
            tstarts, tstops = get_time_intervals(tel_altaz, altmin, altmax, azmin, azmax, accuracy)
            tstops = Time(tstarts + tobs / (len(tstarts)))

        tstarts, tstops = enforce_max_interval_length(tstarts, tstops, max_run_duration)

        if use_wobble:
            pos_angles = wobble_start_angle + np.linspace(0, 2* np.pi, num=wobble_count+1)[:-1] * u.rad

            runs = tuple(
                SkyDataRun(
                    tel_pos.directional_offset_by(pos_angles[run_id % wobble_count], wobble_offset),
                    tstart,
                    tstop,
                    obsloc,
                    run_id
                )
                for run_id, (tstart, tstop) in enumerate(zip(tstarts, tstops))
            )

        else:
            runs = tuple(
                SkyDataRun(tel_pos, tstart, tstop, obsloc, run_id)
                for run_id, (tstart, tstop) in enumerate(zip(tstarts, tstops))
            )
        
        return runs


class DataMatchingGenerator:
    """
    Generator of the sky runs within a given alt/az box
    """
    @classmethod
    def get_runs_from_config(cls, config):
        """
        Returns runs corresponding to the specified config.

        Parameters
        ----------
        config: dict
            Configuration dictionary

        Returns
        -------
        runs: tuple
            Generated runs
        """

        if isinstance(config, str):
            cfg = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        else:
            cfg = config

        obsloc = EarthLocation(
            lat=u.Quantity(cfg['location']['lat']),
            lon=u.Quantity(cfg['location']['lon']),
            height=u.Quantity(cfg['location']['height']),
        )

        return cls.get_runs(cfg['file_mask'], obsloc)

    @classmethod
    def get_runs(cls, file_mask, obsloc):
        """
        Generates runs corresponding to specified observations.

        Parameters
        ----------
        file_mask: str
            observations file mask
        obsloc: EarthLocation
            telescope location

        Returns
        -------
        runs: tuple
            Generated runs
        """

        file_list = glob.glob(file_mask)

        runs_info = functools.reduce(
            lambda container, file_name: container + [cls.runs_info(file_name)],
            file_list,
            []
        )

        runs = [
            SkyDataRun(
                run_info['tel_pos'],
                run_info['tstart'],
                run_info['tstop'],
                obsloc,
                run_info['obs_id'],
            )
            for run_info in runs_info
        ]

        return runs

    @classmethod
    def runs_info(cls, file_name):
        """
        Read the data runs info.

        Parameters
        ----------
        file_name: str
            DL3 observations data run file name

        Returns
        -------
        run_info: dict
            Dictionary with the run ID, ICRS pointing coordinates and start/stop times. 
        """
        obs = read_obs(file_name)

        run_info = dict(
            obs_id = obs.obs_id,
            tel_pos = obs.pointing.fixed_icrs,
            tstart = obs.tstart,
            tstop = obs.tstop,
        )

        return run_info