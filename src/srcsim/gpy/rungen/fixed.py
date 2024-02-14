import yaml
import numpy as np
import astropy.units as u
from scipy.interpolate import CubicSpline
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation

from gammapy.data import PointingMode

from ..run import FixedPointingDataRun
from .helpers import get_trajectory, enforce_max_interval_length


class FixedObsGenerator:
    """
    Generator of the DRIFT sky runs with a fixed alt/az position,
    scanning a certain stripe in RA axis.
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

        ra_width = u.Quantity(cfg['pointing']['width'])
        alt = u.Quantity(cfg['pointing']['center']['alt'])
        tel_pos = SkyCoord(
            u.Quantity(cfg['pointing']['center']['ra']),
            u.Quantity(cfg['pointing']['center']['dec']),
            frame='icrs'
        )

        tstart = Time(cfg['time']['start'])
        duration = u.Quantity(cfg['time']['duration'])
        accuracy = u.Quantity(cfg['time']['accuracy'])
        max_run_duration = u.Quantity(cfg['time']['max_run_duration']) if cfg['time']['max_run_duration'] is not None else None

        obsloc = EarthLocation(
            lat=u.Quantity(cfg['location']['lat']),
            lon=u.Quantity(cfg['location']['lon']),
            height=u.Quantity(cfg['location']['height']),
        )

        return cls.get_runs(obsloc, tel_pos, duration, alt, ra_width, tstart, accuracy, max_run_duration)

    @classmethod
    def get_runs(cls, obsloc, tel_pos, tobs, alt, ra_width, tstart=None, accuracy=1*u.minute, max_run_duration=None):
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
        alt: u.Quantity
            target telescope altitude
        ra_width: u.Quantity
            Observation sky patch width in RA direction.
        tstart: Time
            start time for generated runs
        accuracy: u.Quantity
            trajectory time step to use in calculations
        max_run_duration: u.Quantity
            maximal length of a single run to allow

        Returns
        -------
        runs: tuple
            Generated runs
        """

        tel_pos_trail = SkyCoord(
            tel_pos.icrs.ra + ra_width / 2,
            tel_pos.icrs.dec,
            frame='icrs'
        )
        tel_pos_lead = SkyCoord(
            tel_pos.icrs.ra - ra_width / 2,
            tel_pos.icrs.dec,
            frame='icrs'
        )

        # First pass - first day
        track_trail = get_trajectory(
            tel_pos_trail.icrs,
            tstart,
            tstop=tstart + 1 * u.d,
            time_step=accuracy,
            obsloc=obsloc
        )
        track_lead = get_trajectory(
            tel_pos_lead.icrs,
            tstart,
            tstop=tstart + 1 * u.d,
            time_step=accuracy,
            obsloc=obsloc
        )
        cs_lead = CubicSpline(
            track_lead.obstime.mjd,
            track_lead.alt - alt
        )
        cs_trail = CubicSpline(
            track_trail.obstime.mjd,
            track_trail.alt - alt
        )

        tstarts = Time(cs_lead.roots(extrapolate=False), format='mjd')
        tstops = Time(cs_trail.roots(extrapolate=False), format='mjd')
        
        # Total number of the required observation days
        run_durations = tstops - tstarts
        nsequences = (tobs / np.sum(run_durations)).decompose()
        ndays_full = np.floor(nsequences)
        ndays_total = np.ceil(nsequences)

        if ndays_total > 1:
            # Next pass - to cover the simulation time interval with fully completed runs
            track_lead = get_trajectory(
                tel_pos_lead.icrs,
                tstart,
                tstop=tstart + ndays_full * u.d,
                time_step=accuracy,
                obsloc=obsloc
            )
            track_trail = get_trajectory(
                tel_pos_trail.icrs,
                tstart,
                tstop=tstart + ndays_full * u.d,
                time_step=accuracy,
                obsloc=obsloc
            )

            cs_lead = CubicSpline(
                track_lead.obstime.mjd,
                track_lead.alt - alt
            )
            cs_trail = CubicSpline(
                track_trail.obstime.mjd,
                track_trail.alt - alt
            )
            tstarts = Time(cs_lead.roots(extrapolate=False), format='mjd')
            tstops = Time(cs_trail.roots(extrapolate=False), format='mjd')
            remaining_tobs = tobs - np.sum(tstops - tstarts)
            
            # Next pass - additional incomplete runs
            track_lead = get_trajectory(
                tel_pos_lead.icrs,
                tstart=tstart + ndays_full * u.d,
                tstop=tstart + ndays_total * u.d,
                time_step=accuracy,
                obsloc=obsloc
            )
            cs_lead = CubicSpline(
                track_lead.obstime.mjd,
                track_lead.alt - alt
            )
            _tstarts = Time(cs_lead.roots(extrapolate=False), format='mjd')
            _tstops = Time(_tstarts + remaining_tobs / (len(_tstarts)))

            # Likely an astropy bug (tested with v5.3.4):
            # the latter does not work if len(tstarts) == len(_tstarts)
            # tstarts = Time([tstarts, _tstarts])
            # tstops = Time([tstops, _tstops])
            tstarts = Time(np.concatenate([tstarts.mjd, _tstarts.mjd]), format='mjd')
            tstops = Time(np.concatenate([tstops.mjd, _tstops.mjd]), format='mjd')

        else:
            track_lead = get_trajectory(
                tel_pos_lead.icrs,
                tstart,
                tstop=tstart + ndays_total * u.d,
                time_step=accuracy,
                obsloc=obsloc
            )
            track_trail = get_trajectory(
                tel_pos_trail.icrs,
                tstart,
                tstop=tstart + ndays_total * u.d,
                time_step=accuracy,
                obsloc=obsloc
            )

            cs_lead = CubicSpline(
                track_lead.obstime.mjd,
                track_lead.alt - alt
            )
            cs_trail = CubicSpline(
                track_trail.obstime.mjd,
                track_trail.alt - alt
            )
            tstarts = Time(cs_lead.roots(extrapolate=False), format='mjd')
            tstops = Time(tstarts + tobs / (len(tstarts)))

        # Telescope positions before the time interevals will be sliced
        # These should contain only two different values - those before and after culmination
        _frame = AltAz(
            obstime=tstarts,
            location=obsloc
        )
        _tel_pos_altaz = tel_pos_lead.transform_to(_frame)

        tstarts, tstops = enforce_max_interval_length(tstarts, tstops, max_run_duration)

        # Choosing the closest Alt/Az telescope position from those before interval slicing
        tel_pos_altaz = [
            _tel_pos_altaz[abs(_tel_pos_altaz.obstime - tstart).argmin()]
            for tstart in tstarts
        ]

        runs = tuple(
            FixedPointingDataRun(target_altaz, tstart, tstop, obsloc, run_id, PointingMode.DRIFT)
            for run_id, (tstart, tstop, target_altaz) in enumerate(zip(tstarts, tstops, tel_pos_altaz))
        )

        return runs