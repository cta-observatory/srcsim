import yaml
import numpy as np
import astropy.units as u
from scipy.interpolate import CubicSpline
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation

from ..run import FixedPointingDataRun
from .helpers import get_trajectory, enforce_max_interval_length


class FixedObsGenerator:
    @classmethod
    def get_runs_from_config(cls, config):
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

        tstarts, tstops = enforce_max_interval_length(tstarts, tstops, max_run_duration)

        frame = AltAz(
            obstime=tstarts,
            location=obsloc
        )
        tel_pos_altaz = tel_pos_lead.transform_to(frame)

        runs = tuple(
            FixedPointingDataRun(target_altaz, tstart, tstop, obsloc, run_id)
            for run_id, (tstart, tstop, target_altaz) in enumerate(zip(tstarts, tstops, tel_pos_altaz))
        )

        return runs
