import numpy as np
import pandas as pd
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
        
    def get_events(self, mcevents, source):
        events = []
        nsamples = len(mcevents.samples)
        
        ntimebins = 10
        tedges = np.linspace(self.tstart, self.tstop, num=ntimebins+1)
        tdelta = np.diff(tedges)
        
        for obstime, dt in zip(tedges, tdelta):
            frame = AltAz(obstime=obstime, location=self.obsloc)
            tel_pos = self.tel_pos.transform_to(frame)
        
            for sample in mcevents.samples:
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
                expected_dnde = source.dnde(sample.evt_energy)
                expected_dndo = source.dndo(coords.icrs)
                expected_flux = expected_dnde * expected_dndo

                model_dnde = sample.dnde(sample.evt_energy)
                model_dndo = sample.dndo(sample.evt_coord)
                model_flux = model_dnde * model_dndo

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

                events.append(evt)
            
        events = pd.concat(events)
        
        return events
