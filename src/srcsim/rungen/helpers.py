import numpy as np
from astropy.time import Time
from astropy.coordinates import AltAz


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