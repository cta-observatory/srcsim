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


def get_time_intervals(tel_altaz, altmin, altmax, azmin, azmax, time_step):
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
    
    return tstarts, tstops


class AltAzBoxGenerator:
    def __init__(self, obsloc, tel_pos, tobs, altmin, altmax, azmin, azmax, tstart=None, time_step=1*u.minute):
        self.obsloc = obsloc
        self.tel_pos = tel_pos
        self.tobs = tobs
        self.altmin = altmin
        self.altmax = altmax
        self.azmin = azmin
        self.azmax = azmax
        self.tstart = tstart
        self.time_step = time_step

    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'Obsloc':.<20s}: {self.obsloc}
    {'Tel. pos':.<20s}: {self.tel_pos}
    {'Alt range':.<20s}: [{self.altmin.to('deg').value:.2f} - {self.altmax.to('deg').value:.2f}] deg
    {'Az range':.<20s}: [{self.azmin.to('deg').value:.2f} - {self.azmax.to('deg').value:.2f}] deg
    {'T start':.<20s}: {self.tstart}
    {'T step':.<20s}: {self.time_step}
"""
        )

        return super().__repr__()

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
        tobs = u.Quantity(cfg['time']['duration'])
        time_step = u.Quantity(cfg['time']['step'])
        obsloc = EarthLocation(
            lat=u.Quantity(cfg['location']['lat']),
            lon=u.Quantity(cfg['location']['lon']),
            height=u.Quantity(cfg['location']['height']),
        )

        return cls.get_runs(obsloc, tel_pos, tobs, altmin, altmax, azmin, azmax, tstart, time_step)

    @classmethod
    def get_runs(cls, obsloc, tel_pos, tobs, altmin, altmax, azmin, azmax, tstart=None, time_step=1*u.minute):
        if tstart is None:
            tstart = Time('1970-01-01')

        tel_altaz = get_trajectory(
            tel_pos,
            tstart,
            tstop=tstart + 1 * u.d,
            time_step=time_step,
            obsloc=obsloc
        )
        
        # Second pass with the start on the source anti-culmination
        tstart = tel_altaz.obstime[tel_altaz.alt.argmin()]
        tel_altaz = get_trajectory(
            tel_pos,
            tstart,
            tstop=tstart + 1 * u.d,
            time_step=time_step,
            obsloc=obsloc
        )
        
        tstarts, tstops = get_time_intervals(tel_altaz, altmin, altmax, azmin, azmax, time_step)
        run_durations = tstops - tstarts
        
        nsequences = (tobs / np.sum(run_durations)).decompose()
        ndays_full = np.floor(nsequences)
        ndays_total = np.ceil(nsequences)
        
        # Third pass - to cover the simulation time interval with fully completed runs
        tel_altaz = get_trajectory(
            tel_pos,
            tstart,
            tstop=tstart + ndays_full * u.d,
            time_step=time_step,
            obsloc=obsloc
        )
        tstarts, tstops = get_time_intervals(tel_altaz, altmin, altmax, azmin, azmax, time_step)
        remaining_tobs = tobs - np.sum(tstops - tstarts)
        
        # Fourth pass - additional incomplete runs
        _tel_altaz = get_trajectory(
            tel_pos,
            tstart=tstart + ndays_full * u.d,
            tstop=tstart + ndays_total * u.d,
            time_step=time_step,
            obsloc=obsloc
        )
        _tstarts, _tstops = get_time_intervals(_tel_altaz, altmin, altmax, azmin, azmax, time_step)
        _tstops = Time(_tstarts + remaining_tobs / (len(_tstarts)))
        
        tstarts = Time([tstarts, _tstarts])
        tstops = Time([tstops, _tstops])
        
        runs = tuple(
            DataRun(tel_pos, tstart, tstop, obsloc)
            for tstart, tstop in zip(tstarts, tstops)
        )
        
        return runs
