import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, SkyOffsetFrame, AltAz


class DataRun:
    def __init__(self, tel_pos, tstart, tstop, obsloc, id=0):
        self.id = id
        self.tel_pos = tel_pos
        self.obsloc = obsloc
        self.tstart = tstart
        self.tstop = tstop 

    @classmethod
    def from_config(cls, config):
        pass

    def to_dict(self):
        pass

    def tel_pos_to_altaz(self, frame):
        pass

    @classmethod
    def time_sort(cls, events):
        timestamp = np.array(events["trigger_time"])
        sorting = np.argsort(timestamp)

        return events.iloc[sorting]

    @classmethod
    def update_time_delta(cls, events):
        event_time = events['trigger_time'].to_numpy()
        delta_time = np.zeros_like(event_time)

        if len(event_time) > 0:
            argsorted = event_time.argsort()
            dt = np.diff(event_time[argsorted])
            delta_time[argsorted[:-1]] = dt
            delta_time[argsorted[-1]] = dt[-1]

        events = events.drop(
            columns=['delta_t'],
            errors='ignore'
        )
        events = events.assign(
            delta_t = delta_time,
        )

        return events

    def predict(self, mccollections, source, tel_pos_tolerance=None, time_step=1*u.minute):
        events = []

        unix_edges = np.arange(self.tstart.unix, self.tstop.unix, step=time_step.to('s').value)
        if unix_edges[-1] < self.tstop.unix:
            unix_edges = np.concatenate((unix_edges, [self.tstop.unix]))

        tedges = Time(unix_edges, format='unix')

        if len(tedges) > 1:
            tdelta = np.diff(tedges)
        else:
            tdelta = [self.tstop - self.tstart]

        for tstart, dt in zip(tedges[:-1], tdelta):
            frame = AltAz(
                obstime=tstart + dt/2,
                location=self.obsloc
            )
            tel_pos = self.tel_pos_to_altaz(frame)

            if tel_pos_tolerance is None:
                mc = mccollections[source.emission_type].get_closest(tel_pos.altaz)
            elif isinstance(tel_pos_tolerance, u.Quantity):
                mc = mccollections[source.emission_type].get_nearby(tel_pos, tel_pos_tolerance)
            elif isinstance(tel_pos_tolerance, list) or isinstance(tel_pos_tolerance, tuple):
                mc = mccollections[source.emission_type].get_in_box(
                    tel_pos,
                    max_lon_offset=tel_pos_tolerance[0],
                    max_lat_offset=tel_pos_tolerance[1],
                )
            else:
                raise ValueError(f"Data type '{type(tel_pos_tolerance)}' for argument 'tel_pos_tolerance' is not supported")

            nsamples = len(mc.samples)

            for sample in mc.samples:
                # Randomly distributing events within the time bin
                arrival_time = np.random.uniform(tstart.unix, (tstart+dt).unix, size=len(sample.data_table))

                # Recalculating telescope Alt/Az for the mock event arrival times
                current_frame = AltAz(
                    obstime=Time(arrival_time, format='unix'),
                    location=self.obsloc
                )
                current_tel_pos = self.tel_pos_to_altaz(current_frame)

                # Astropy does not pass the location / time
                # to the offset frame, need to do this manually
                offset_frame = SkyOffsetFrame(
                    origin=current_tel_pos.altaz.skyoffset_frame().origin,
                    location=current_tel_pos.altaz.frame.location,
                    obstime=current_tel_pos.altaz.frame.obstime
                )
                coords = SkyCoord(
                    sample.evt_coord.skyoffsetaltaz.lon,
                    sample.evt_coord.skyoffsetaltaz.lat,
                    frame=offset_frame
                )
                expected_flux = source.dndedo(sample.evt_energy, coords.icrs)
                model_flux = sample.dndedo(sample.evt_energy, sample.evt_coord)

                weights = (1 / nsamples * dt * expected_flux / model_flux).decompose()

                n_mc_events = len(sample.evt_energy)
                n_events = np.random.poisson(weights.sum())

                if n_events > 0:
                    p = weights / weights.sum()
                    idx = np.random.choice(
                        np.arange(n_mc_events),
                        size=n_events,
                        p=p
                    )

                    evt = sample.data_table.iloc[idx]
                    offset_frame = offset_frame[idx]
                    arrival_time = arrival_time[idx]
                    current_tel_pos = current_tel_pos[idx]

                    # Dropping the columns we're going to (re-)fill
                    evt = evt.drop(
                        columns=['dragon_time', 'trigger_time'],
                        errors='ignore'
                    )
                    evt = evt.drop(
                        columns=['mc_az_tel', 'mc_alt_tel', 'az_tel', 'alt_tel', 'ra_tel', 'dec_tel'],
                        errors='ignore'
                    )
                    evt = evt.drop(
                        columns=['reco_az', 'reco_alt', 'reco_ra', 'reco_dec'],
                        errors='ignore'
                    )

                    # Events arrival time
                    evt = evt.assign(
                        dragon_time = arrival_time,
                        trigger_time = arrival_time,
                    )

                    # Telescope pointing
                    evt = evt.assign(
                        mc_az_tel = current_tel_pos.az.to('rad').value,
                        mc_alt_tel = current_tel_pos.alt.to('rad').value,
                        az_tel = current_tel_pos.az.to('rad').value,
                        alt_tel = current_tel_pos.alt.to('rad').value,
                        ra_tel = current_tel_pos.icrs.ra.to('rad').value,
                        dec_tel = current_tel_pos.icrs.dec.to('rad').value
                    )

                    # Reconstructed events coordinates
                    reco_coords = SkyCoord(
                        evt['reco_src_x'].to_numpy() * sample.units['distance'] * sample.cam2angle,
                        evt['reco_src_y'].to_numpy() * sample.units['distance'] * sample.cam2angle,
                        frame=offset_frame
                    )
                    evt = evt.assign(
                        reco_az = reco_coords.altaz.az.to('rad').value,
                        reco_alt = reco_coords.altaz.alt.to('rad').value,
                        reco_ra = reco_coords.icrs.ra.to('rad').value,
                        reco_dec = reco_coords.icrs.dec.to('rad').value,
                    )
                else:
                    evt = sample.data_table.iloc[0:0]
                    evt = evt.assign(
                        dragon_time = np.zeros(0),
                        trigger_time = np.zeros(0),
                    )
                    evt = evt.assign(
                        mc_az_tel = np.zeros(0),
                        mc_alt_tel = np.zeros(0),
                        az_tel = np.zeros(0),
                        alt_tel = np.zeros(0),
                        ra_tel = np.zeros(0),
                        dec_tel = np.zeros(0)
                    )
                    evt = evt.assign(
                        reco_az = np.zeros(0),
                        reco_alt = np.zeros(0),
                        reco_ra = np.zeros(0),
                        reco_dec = np.zeros(0)
                    )

                evt = evt.drop('mc_src_name', errors='ignore')
                evt = evt.assign(
                    mc_src_name = np.repeat(source.name, len(evt))
                )

                events.append(evt)

        events = pd.concat(events)

        events = self.update_time_delta(events)

        return events
