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
    {'Tstart':.<20s}: {self.tstart}
    {'Tstop':.<20s}: {self.tstop}
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

    def predict(self, mccollection, source, tel_pos_tolerance=None):
        events = []
        
        time_step = 1 * u.minute
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
                mc = mccollection.get_closest(tel_pos.altaz)
            else:
                mc = mccollection.get_nearby(tel_pos, tel_pos_tolerance)

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

                evt.loc[slice(None), 'mc_az_tel'] = tel_pos.az.to('rad').value
                evt.loc[slice(None), 'mc_alt_tel'] = tel_pos.alt.to('rad').value

                evt = evt.assign(dragon_time = np.linspace(tstart.unix, (tstart+dt).unix, num=len(evt)))

                events.append(evt)
            
        events = pd.concat(events)
        
        return events
