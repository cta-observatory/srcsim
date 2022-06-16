import yaml
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation

from .run import DataRun


def get_trajectory(tel_pos, tstart, tstop, time_step, obsloc):
    times = Time(
        np.arange(tstart.unix, tstop.unix, step=time_step.to('s').value),
        format='unix'
    )
    frame = AltAz(obstime=times, location=obsloc)
    tel_altaz = tel_pos.transform_to(frame)
    
    return tel_altaz


def enforce_max_interval_length(tstarts, tstops, max_length):
    tstarts_new = []
    tstops_new = []

    for tstart, tstop in zip(tstarts, tstops):
        interval_duration = tstop - tstart

        if interval_duration > max_length:
            time_edges = Time(
                np.arange(tstart.unix, tstop.unix, step=max_length.to('s').value),
                format='unix'
            )
            if tstop not in time_edges:
                time_edges = Time([time_edges, tstop])

            for tmin, tmax in zip(time_edges[:-1], time_edges[1:]):
                tstarts_new.append(tmin)
                tstops_new.append(tmax)

        else:
            tstarts_new.append(tstart)
            tstops_new.append(tstop)

    return Time(tstarts_new), Time(tstops_new)


def get_time_intervals(tel_altaz, altmin, altmax, azmin, azmax, time_step, max_duration=None):
    in_box = (tel_altaz.az > azmin) & (tel_altaz.az <= azmax) & (tel_altaz.alt > altmin) & (tel_altaz.alt < altmax)
    nodes = np.where(np.diff(tel_altaz.obstime[in_box].unix) > time_step.to('s').value)[0]
    
    starts = np.concatenate(([0], nodes+1))
    
    if (len(tel_altaz.obstime[in_box]) - 1) not in nodes:
        stops = np.concatenate((nodes, [len(tel_altaz.obstime[in_box]) - 1]))
    else:
        stops = nodes
        
    intervals = tuple((start, stop) for start, stop in zip(starts, stops))

    tstarts = Time([tel_altaz.obstime[in_box][interval[0]] for interval in intervals])
    tstops = Time([tel_altaz.obstime[in_box][interval[1]] for interval in intervals])

    if max_duration is not None:
        tstarts, tstops = enforce_max_interval_length(tstarts, tstops, max_duration)
    
    return tstarts, tstops


class AltAzBoxGenerator:
    @classmethod
    def get_runs_from_config(cls, config):
        if isinstance(config, str):
            cfg = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        else:
            cfg = config
        
        tel_pos = SkyCoord(
            u.Quantity(cfg['pointing']['ra']),
            u.Quantity(cfg['pointing']['dec']),
            frame='icrs'
        )

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

        return cls.get_runs(obsloc, tel_pos, duration, altmin, altmax, azmin, azmax, tstart, accuracy, max_run_duration)

    @classmethod
    def get_runs(cls, obsloc, tel_pos, tobs, altmin, altmax, azmin, azmax, tstart=None, accuracy=1*u.minute, max_run_duration=None):
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
        
        tstarts, tstops = get_time_intervals(tel_altaz, altmin, altmax, azmin, azmax, accuracy, max_run_duration)
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
            tstarts, tstops = get_time_intervals(tel_altaz, altmin, altmax, azmin, azmax, accuracy, max_run_duration)
            remaining_tobs = tobs - np.sum(tstops - tstarts)
        
            # Fourth pass - additional incomplete runs
            _tel_altaz = get_trajectory(
                tel_pos,
                tstart=tstart + ndays_full * u.d,
                tstop=tstart + ndays_total * u.d,
                time_step=accuracy,
                obsloc=obsloc
            )
            _tstarts, _tstops = get_time_intervals(_tel_altaz, altmin, altmax, azmin, azmax, accuracy, max_run_duration)
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
            tstarts, tstops = get_time_intervals(tel_altaz, altmin, altmax, azmin, azmax, accuracy, max_run_duration)
            tstops = Time(tstarts + tobs / (len(tstarts)))

        runs = tuple(
            DataRun(tel_pos, tstart, tstop, obsloc, run_id)
            for run_id, (tstart, tstop) in enumerate(zip(tstarts, tstops))
        )
        
        return runs
