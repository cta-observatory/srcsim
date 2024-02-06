import numpy as np
from astropy.time import Time
from astropy.coordinates import AltAz

from gammapy.irf import load_irf_dict_from_file
from gammapy.data import Observation
from gammapy.data.event_list import EventList
from gammapy.data.gti import GTI
from gammapy.data.pointing import FixedPointingInfo


def get_trajectory(tel_pos, tstart, tstop, time_step, obsloc):
    """
    Compute telescope alt/az trajctory.

    Parameters
    ----------
    tel_pos: SkyCoord
        Equatorial telscope pointing coordinates
    tstart: Time
        Start moment of the trajectory
    tstop: Time
        Stop moment of the trajectory
    time_step: u.Quantity
        Trajectory time step to use
    obsloc: EarthLocation
        Telescope location

    Returns
    -------
    tel_altaz: SkyCoord
        Computed alt/az trajectory
    """

    times = Time(
        np.arange(tstart.unix, tstop.unix, step=time_step.to('s').value),
        format='unix'
    )
    frame = AltAz(obstime=times, location=obsloc)
    tel_altaz = tel_pos.transform_to(frame)
    
    return tel_altaz


def enforce_max_interval_length(tstarts, tstops, max_length):
    """
    Enforcing the input time intervals do not exceed 
    max_length duration and splits them otherwise.

    Parameters
    ----------
    tstarts: Time
        Start moments of the input time intervals
    tstops: Time
        Stop moments of the input time intervals
    max_length: u.Quantity
        Maximal length of a single time interval to allow

    Returns
    -------
    tstarts: Time
        Start moments of the enforced intervals
    tstops: Time
        Stop moments of the enforced intervals
    """

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
    """
    Compute time intervals when a given alt/az trajectory is contained within
    the specified alt/az box.

    Parameters
    ----------
    tel_altaz: SkyCoord 
        Array-like alt/az coordinates of the trajectory to use.
        Should contain both coordinates and obstime.
    altmin: u.Quantity
        Minimal trajectory altitude to use.
    altmax: u.Quantity
        Maximal trajectory altitude to use.
    azmin: u.Quantity
        Minimal trajectory azimuth to use.
    azmax: u.Quantity
        Maximal trajectory azimuth to use.
    time_step: u.Quantity
        Minimal time interval to assume between the subsequent time intervals
    max_duration: u.Quantity
        Maximal duration of a single time interval to allow

    Returns
    -------
    tstarts: Time
        Start moments of the identified intervals
    tstops: Time
        Stop moments of the identified intervals
    """

    in_box = (tel_altaz.az > azmin) & (tel_altaz.az <= azmax) & (tel_altaz.alt > altmin) & (tel_altaz.alt < altmax)
    nodes = np.where(np.diff(tel_altaz.obstime[in_box].unix) > time_step.to('s').value)[0]
    
    starts = np.concatenate(([0], nodes+1))
    
    if (len(tel_altaz.obstime[in_box]) - 1) not in nodes:
        stops = np.concatenate((nodes, [len(tel_altaz.obstime[in_box]) - 1]))
    else:
        stops = nodes
        
    intervals = tuple((start, stop) for start, stop in zip(starts, stops))

    # TODO: check if sum(in_box) > 0 and len(intervals) > 0

    tstarts = Time([tel_altaz.obstime[in_box][interval[0]] for interval in intervals])
    tstops = Time([tel_altaz.obstime[in_box][interval[1]] for interval in intervals])

    if max_duration is not None:
        tstarts, tstops = enforce_max_interval_length(tstarts, tstops, max_duration)
    
    return tstarts, tstops


def read_obs(event_file, irf_file=None):
    """Create an Observation from a Event List and an (optional) IRF file.

    This is a modified version of gammapy.data.observations.Observation.read()
    In gammapy 1.1, `FixedPointingInfo.from_fits_header()` [1] mistakenly 
    keeps ALT_PNT and AZ_PNT as `str` without converting to floats / quantities; 
    their further multiplication with `u.deg` leads to `astropy.units.core.Unit` type 
    instead of `astropy.units.quantity.Quantity` - incompatible with `AltAz()` constructor.

    Parameters
    ----------
    event_file : str, Path
        path to the .fits file containing the event list and the GTI
    irf_file : str, Path
        (optional) path to the .fits file containing the IRF components,
        if not provided the IRF will be read from the event file

    Returns
    -------
    observation : `~gammapy.data.Observation`
        observation with the events and the irf read from the file

    References
    ----------
    [1] https://docs.gammapy.org/1.1/_modules/gammapy/data/pointing.html#FixedPointingInfo
    """
    from gammapy.irf.io import load_irf_dict_from_file

    events = EventList.read(event_file)

    gti = GTI.read(event_file)

    irf_file = irf_file if irf_file is not None else event_file
    irf_dict = load_irf_dict_from_file(irf_file)

    obs_info = events.table.meta

    # Removing problematic keys
    del obs_info['AZ_PNT']
    del obs_info['ALT_PNT']
    
    return Observation(
        events=events,
        gti=gti,
        obs_info=obs_info,
        obs_id=obs_info.get("OBS_ID"),
        pointing=FixedPointingInfo.from_fits_header(obs_info),
        **irf_dict,
    )
