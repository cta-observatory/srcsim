import astropy.units as u
from astropy.time import Time
from astropy.coordinates import AltAz

from gammapy.data import Observation
from gammapy.datasets import MapDataset
from gammapy.maps import MapAxis, WcsGeom
# from gammapy.datasets import MapDatasetEventSampler
# from gammapy.makers import MapDatasetMaker

from .makers import MapDatasetMaker
from .simulate import MapDatasetEventSampler


class DataRun:
    mode = None

    def __init__(self, tel_pos, tstart, tstop, obsloc, id=0):
        """
        Creates a generic data run class instance.

        Parameters
        ----------
        tel_pos: astropy.coordinates.SkyCoord
            Telescope pointing position
        tstart: astropy.time.Time
            Run time start
        tstop: astropy.time.Time
            Run time stop
        obsloc: astropy.coordinates.EarthLocation
            Telescope location
        id: int
            Run ID
        """
        self.id = id
        self.tel_pos = tel_pos
        self.obsloc = obsloc
        self.tstart = tstart
        self.tstop = tstop 

    @classmethod
    def from_config(cls, config):
        """
        Create run from the specified configuration.
        This method needs to be overloaded in the child classes.

        Parameters
        ----------
        config: str or dict
            Run configuration to use. If string,
            configuration will be loaded from the YAML
            file specified by "config".

        Returns
        -------
        run: DataRun
            Corresponding DataRun child instance

        Notes
        -----
        See also self.to_dict()
        """
        pass

    @property
    def pointing(self):
        """
        Observation pointing info.
        This method needs to be overloaded in the child classes.

        Returns
        -------
        info: gammapy.data.FixedPointingInfo
        """
        return None

    @property
    def slew_length_ra(self):
        """
        Observation slew distance in RA.
        Will be used to define the simulation WCS extension.

        Returns
        -------
        u.Quantity
            slew length in RA
        """
        return 0 * u.deg

    @property
    def slew_length_dec(self):
        """
        Observation slew distance in Dec.
        Will be used to define the simulation WCS extension.

        Returns
        -------
        u.Quantity
            slew length in Dec
        """
        return 0 * u.deg

    @property
    def tel_pos_center_icrs(self):
        """
        Telescope pointing center in equatorial (ICRS) coordinates

        Returns
        -------
        astropy.coordinates.SkyCoord
            pointing center
        """
        return None

    def to_dict(self):
        """
        Converts the class definition to a configuration dict.

        Returns
        -------
        dict:
            Class configuration as dict

        Notes
        -----
        See also self.from_config()
        """
        pass

    def tel_pos_to_altaz(self, frame):
        """
        Transform ICRS telescope poiting to the specified alt/az frame.

        Parameters
        ----------
        frame: astropy.coordinates.AltAz
            Frame to transform the telescope pointing to.

        Returns
        -------
        astropy.coordinates.SkyCoord
            Telescope pointing in alt/az frame
        """
        pass

    def predict(self, irf_collection, model, tel_pos_tolerance=None):
        """
        Creates an observation that would correspond to this run
        given the specified IRFs and field of view model.

        Parameters
        ----------
        irf_collection: .irf.IRFCollection
            A collection to choose the appropriate IRF from
        model: gammapy.modeling.models.Model
            Field of view model
        tel_pos_tolerance: None, list-like or u.Quantity
            Position difference between IRFs and telescope pointing to observe
            - None: use the closest IRF in collection
            - u.Quantity: select IRFs within a circle of the corresponding radius
            - list or tuple: select IRFs within the corresponding box on the sky
            NOTE: only None option is presently implemented
        """
        frame_tref = AltAz(
            obstime=Time(
                (self.tstart.mjd + self.tstop.mjd) / 2,
                format='mjd'
            ),
            location=self.obsloc
        )

        tel_pos = self.tel_pos_to_altaz(frame_tref)

        if tel_pos_tolerance is None:
            irfs = irf_collection.get_closest(tel_pos.altaz)
        elif isinstance(tel_pos_tolerance, u.Quantity):
            irfs = irf_collection.get_nearby(tel_pos, tel_pos_tolerance)
        elif isinstance(tel_pos_tolerance, list) or isinstance(tel_pos_tolerance, tuple):
            irfs = irf_collection.get_in_box(
                tel_pos,
                max_lon_offset=tel_pos_tolerance[0],
                max_lat_offset=tel_pos_tolerance[1],
            )
        else:
            raise ValueError(f"Data type '{type(tel_pos_tolerance)}' for argument 'tel_pos_tolerance' is not supported")

        observation = Observation.create(
            pointing=self.pointing,
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
            nbin=20,
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
            skydir=self.tel_pos_center_icrs,
            width=(
                self.slew_length_ra.to('deg').value + 5,
                self.slew_length_dec.to('deg').value + 5)
            ,
            binsz=0.1,
            frame="icrs",
            axes=[energy_axis],
        )

        empty = MapDataset.create(
            geom,
            energy_axis_true=energy_axis_true,
            migra_axis=migra_axis,
            name="sim-dataset",
        )

        maker = MapDatasetMaker(
            selection=["exposure", "background", "psf", "edisp"]
        )
        dataset = maker.run(empty, observation)

        dataset.models = model

        sampler = MapDatasetEventSampler(random_state=0)
        events = sampler.run(dataset, observation)

        observation._events = events

        return observation
