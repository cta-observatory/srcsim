import yaml
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, SkyOffsetFrame, EarthLocation, AltAz


class DataRun:
    def __init__(self, tel_pos, tstart, tstop, obsloc, id=0):
        self.id = id
        self.tel_pos = tel_pos
        self.obsloc = obsloc
        self.tstart = tstart
        self.tstop = tstop
        
    def __repr__(self):
        frame_start = AltAz(obstime=self.tstart, location=self.obsloc)
        frame_stop = AltAz(obstime=self.tstop, location=self.obsloc)
        print(
f"""{type(self).__name__} instance
    {'ID':.<20s}: {self.id}
    {'Tel. RA/Dec':.<20s}: {self.tel_pos}
    {'Tstart':.<20s}: {self.tstart.isot}
    {'Tstop':.<20s}: {self.tstop.isot}
    {'Tel. azimuth':.<20s}: [{self.tel_pos.transform_to(frame_start).az.to('deg'):.2f} - {self.tel_pos.transform_to(frame_stop).az.to('deg'):.2f}]
    {'Tel. alt':.<20s}: [{self.tel_pos.transform_to(frame_start).alt.to('deg'):.2f} - {self.tel_pos.transform_to(frame_stop).alt.to('deg'):.2f}]
"""
        )

        return super().__repr__() 

    @classmethod
    def from_config(cls, config):
        if isinstance(config, str):
            cfg = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        else:
            cfg = config

        data_run = cls(
            SkyCoord(
                u.Quantity(cfg['pointing']['ra']),
                u.Quantity(cfg['pointing']['dec']),
                frame='icrs'
            ),
            Time(cfg['time']['start']),
            Time(cfg['time']['stop']),
            EarthLocation(
                lat=u.Quantity(cfg['location']['lat']),
                lon=u.Quantity(cfg['location']['lon']),
                height=u.Quantity(cfg['location']['height']),
            ),
            cfg['id'] if 'id' in cfg else 0
        )

        return data_run

    @classmethod
    def time_sort(cls, events):
        timestamp = np.array(events["trigger_time"])
        sorting = np.argsort(timestamp)

        return events.iloc[sorting]

    @classmethod
    def update_time_delta(cls, events):
        event_time = events['trigger_time'].to_numpy()
        delta_time = np.zeros_like(event_time)

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

    def to_dict(self):
        data = {'id': self.id, 'pointing': {}, 'time': {}, 'location': {}}

        data['pointing']['ra'] = str(self.tel_pos.icrs.ra.to('deg').value) + ' deg'
        data['pointing']['dec'] = str(self.tel_pos.icrs.dec.to('deg').value) + ' deg'

        data['time']['start'] = self.tstart.isot
        data['time']['stop'] = self.tstop.isot

        data['location']['lon'] = str(self.obsloc.lon.to('deg').value) + ' deg'
        data['location']['lat'] = str(self.obsloc.lat.to('deg').value) + ' deg'
        data['location']['height'] = str(self.obsloc.height.to('m').to_string())

        return data

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
            tel_pos = self.tel_pos.transform_to(frame)
        
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
                current_tel_pos = self.tel_pos.transform_to(current_frame)

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

                # Per-row weights (no copy of data_table)
                row_weights = (1 / nsamples * dt * expected_flux / model_flux).decompose().value

                df = sample.data_table

                # groupby.indices gives: {(obs_id, event_id): array([row indices])}
                event_groups = df.groupby(['obs_id', 'event_id']).indices

                # row_weights: NumPy array aligned with df rows
                event_keys = list(event_groups.keys())
                event_weights = np.fromiter(
                    (row_weights[event_groups[k]].sum() for k in event_keys),
                    dtype=float,
                    count=len(event_keys)
                )

                # Poisson draw
                n_events = np.random.poisson(event_weights.sum())

                if n_events > 0:
                    p = event_weights / event_weights.sum()

                    # Sample events
                    idx_evt = np.random.choice(
                        np.arange(len(event_keys)),
                        size=n_events,
                        p=p
                    )

                    # Count copies per event
                    copy_counts = {}
                    for i in idx_evt:
                        copy_counts[i] = copy_counts.get(i, 0) + 1
                    
                    evt_indices, evt_copy_counts = np.unique(idx_evt, return_counts=True)
                    rows_per_event = np.fromiter(
                        (len(event_groups[event_keys[i]]) for i in evt_indices),
                        dtype=int,
                        count=len(evt_indices)
                    )
                    # Concatenate row indices per event
                    rows_flat = np.concatenate(
                        [event_groups[event_keys[i]] for i in evt_indices]
                    )
                    # Repeat each event's rows according to copy count
                    row_idx = np.repeat(
                        rows_flat,
                        np.repeat(evt_copy_counts, rows_per_event)
                    )
                    event_copy_id = np.concatenate([
                        np.repeat(np.arange(n), rows_per_event[i])
                        for i, n in enumerate(evt_copy_counts)
                    ])

                    evt = df.iloc[row_idx].assign(event_copy_id=event_copy_id)
                    offset_frame = offset_frame[row_idx]
                    arrival_time = arrival_time[row_idx]
                    current_tel_pos = current_tel_pos[row_idx]
                    coords = coords[row_idx]

                else:
                    # Empty but schema-consistent
                    evt = df.iloc[0:0].assign(event_copy_id=np.zeros(0, dtype=int))
                    offset_frame = offset_frame[0:0]
                    arrival_time = arrival_time[0:0]
                    current_tel_pos = current_tel_pos[0:0]
                    coords = coords[0:0]

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
                    columns=['true_az', 'true_alt'],
                    errors='ignore'
                )
                evt = evt.drop(
                    columns=['reco_az', 'reco_alt', 'reco_ra', 'reco_dec'],
                    errors='ignore'
                )

                if n_events > 0:
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
                        ra_tel = self.tel_pos.icrs.ra.to('rad').value,
                        dec_tel = self.tel_pos.icrs.dec.to('rad').value
                    )

                    # True events coordinates
                    evt = evt.assign(
                        true_az = coords.altaz.az.to('rad').value,
                        true_alt = coords.altaz.alt.to('rad').value,
                    )

                    # Reconstructed events coordinates
                    if 'reco_src_x' in evt.columns:
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
                    if 'reco_src_x' in evt.columns:
                        evt = evt.assign(
                            reco_az = np.zeros(0),
                            reco_alt = np.zeros(0),
                            reco_ra = np.zeros(0),
                            reco_dec = np.zeros(0)
                        )

                events.append(evt)

        events = pd.concat(events)

        events = self.update_time_delta(events)

        return events
