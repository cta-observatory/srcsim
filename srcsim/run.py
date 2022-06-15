import yaml
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, SkyOffsetFrame, EarthLocation, AltAz


class DataRun:
    def __init__(self, tel_pos, tstart, tstop, obsloc):
        self.tel_pos = tel_pos
        self.obsloc = obsloc
        self.tstart = tstart
        self.tstop = tstop
        
    def __repr__(self):
        frame_start = AltAz(obstime=self.tstart, location=self.obsloc)
        frame_stop = AltAz(obstime=self.tstop, location=self.obsloc)
        print(
f"""{type(self).__name__} instance
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
            )
        )

        return data_run

    def to_dict(self):
        data = {'pointing': {}, 'time': {}, 'location': {}}

        data['pointing']['ra'] = str(self.tel_pos.icrs.ra.to('deg').to_string())
        data['pointing']['dec'] = str(self.tel_pos.icrs.dec.to('deg').to_string())

        data['time']['start'] = self.tstart.isot
        data['time']['stop'] = self.tstop.isot

        data['location']['lon'] = str(self.obsloc.lon.to('deg').to_string())
        data['location']['lat'] = str(self.obsloc.lat.to('deg').to_string())
        data['location']['height'] = str(self.obsloc.height.to('m').to_string())

        return data

    def predict(self, mccollections, source, tel_pos_tolerance=None, time_step=1*u.minute):
        events = []

        tedges = Time(
            np.arange(self.tstart.unix, self.tstop.unix, step=time_step.to('s').value),
            format='unix'
        )
        if len(tedges) > 1:
            tdelta = np.diff(tedges)
        else:
            tdelta = [self.tstop - self.tstart]
        
        for tstart, dt in zip(tedges[:-1], tdelta):
            frame = AltAz(obstime=tstart, location=self.obsloc)
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
                # Astropy does not pass the location / time
                # to the offset frame, need to do this manually
                offset_frame = SkyOffsetFrame(
                    origin=tel_pos.altaz.skyoffset_frame().origin,
                    location=tel_pos.altaz.frame.location,
                    obstime=tel_pos.altaz.frame.obstime
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
                n_events = int(np.round(weights.sum()))
                p = weights / weights.sum()
                idx = np.random.choice(
                    np.arange(n_mc_events),
                    size=n_events,
                    p=p
                )

                evt = sample.data_table.iloc[idx]

                # Events arrival time
                evt = evt.assign(
                    dragon_time = np.linspace(tstart.unix, (tstart+dt).unix, num=len(evt))
                )

                # Telescope pointing
                evt = evt.drop(
                    columns=['mc_az_tel', 'mc_alt_tel', 'az_tel', 'alt_tel'],
                    errors='ignore'
                )
                evt = evt.assign(
                    mc_az_tel = tel_pos.az.to('rad').value,
                    mc_alt_tel = tel_pos.alt.to('rad').value,
                    az_tel = tel_pos.az.to('rad').value,
                    alt_tel = tel_pos.alt.to('rad').value
                )

                # Reconstructed events coordinates
                reco_coords = SkyCoord(
                    evt['reco_src_x'].to_numpy() * sample.units['distance'] * sample.cam2angle,
                    evt['reco_src_y'].to_numpy() * sample.units['distance'] * sample.cam2angle,
                    frame=offset_frame
                )
                evt = evt.drop(columns=['reco_az', 'reco_alt'], errors='ignore')
                evt = evt.assign(
                    reco_az = reco_coords.altaz.az.to('rad').value,
                    reco_alt = reco_coords.altaz.alt.to('rad').value,
                )

                events.append(evt)
            
        events = pd.concat(events)
        
        return events
