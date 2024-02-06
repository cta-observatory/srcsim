import yaml
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from gammapy.data import Observation, PointingMode, FixedPointingInfo
from gammapy.datasets import MapDataset, MapDatasetEventSampler
from gammapy.makers import MapDatasetMaker
from gammapy.maps import MapAxis, WcsGeom

from .base import DataRun


class SkyDataRun(DataRun):
    def __str__(self):
        frame_start = AltAz(obstime=self.tstart, location=self.obsloc)
        frame_stop = AltAz(obstime=self.tstop, location=self.obsloc)
        desc = f"""{type(self).__name__} instance
    {'ID':.<20s}: {self.id}
    {'Tel. RA/Dec':.<20s}: {self.tel_pos}
    {'Tstart':.<20s}: {self.tstart.isot}
    {'Tstop':.<20s}: {self.tstop.isot}
    {'Tel. azimuth':.<20s}: [{self.tel_pos.transform_to(frame_start).az.to('deg'):.2f} - {self.tel_pos.transform_to(frame_stop).az.to('deg'):.2f}]
    {'Tel. alt':.<20s}: [{self.tel_pos.transform_to(frame_start).alt.to('deg'):.2f} - {self.tel_pos.transform_to(frame_stop).alt.to('deg'):.2f}]
"""
        return desc

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
    
    def predict(self, irfs, models, tel_pos_tolerance=None):
        frame_tref = AltAz(
            obstime=Time(
                (self.tstart.mjd + self.tstop.mjd) / 2,
                format='mjd'
            ),
            location=self.obsloc
        )

        tel_pos = self.tel_pos.transform_to(frame_tref)
        
        if tel_pos_tolerance is None:
            irfs = irfs.get_closest(tel_pos.altaz)
        elif isinstance(tel_pos_tolerance, u.Quantity):
            irfs = irfs.get_nearby(tel_pos, tel_pos_tolerance)
        elif isinstance(tel_pos_tolerance, list) or isinstance(tel_pos_tolerance, tuple):
            irfs = irfs.get_in_box(
                tel_pos,
                max_lon_offset=tel_pos_tolerance[0],
                max_lat_offset=tel_pos_tolerance[1],
            )
        else:
            raise ValueError(f"Data type '{type(tel_pos_tolerance)}' for argument 'tel_pos_tolerance' is not supported")

        pointing = FixedPointingInfo(fixed_icrs=self.tel_pos, mode=PointingMode.POINTING)

        observation = Observation.create(
            pointing=pointing,
            location=self.obsloc,
            obs_id=self.id,
            tstart=self.tstart,
            tstop=self.tstop,
            irfs=irfs.samples[0].to_dict()
        )
        observation.aeff.meta["TELESCOP"] = 'cta_north'

        energy_axis = MapAxis.from_energy_bounds(
            "0.03 TeV",
            "100 TeV",
            nbin=20,
            per_decade=True
        )
        energy_axis_true = MapAxis.from_energy_bounds(
            "0.01 TeV",
            "300 TeV",
            nbin=30,
            per_decade=True,
            name="energy_true"
        )
        migra_axis = MapAxis.from_bounds(
            0.5,
            2,
            nbin=150,
            node_type="edges",
            name="migra"
        )

        geom = WcsGeom.create(
            skydir=observation.pointing.fixed_icrs,
            width=(5, 5),
            binsz=0.1,
            frame="icrs",
            axes=[energy_axis],
        )

        empty = MapDataset.create(
            geom,
            energy_axis_true=energy_axis_true,
            migra_axis=migra_axis,
            name="my-dataset",
        )

        maker = MapDatasetMaker(
            selection=["exposure", "background", "psf", "edisp"]
        )
        dataset = maker.run(empty, observation)

        dataset.models = models

        sampler = MapDatasetEventSampler(random_state=0)
        events = sampler.run(dataset, observation)

        observation._events = events

        return observation
