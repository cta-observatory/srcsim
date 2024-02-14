import numpy as np
import gammapy

from astropy.time import Time
from gammapy.data import EventList, observatory_locations, PointingMode
from gammapy.modeling.models import (
    ConstantTemporalModel,
)

from gammapy.datasets import MapDatasetEventSampler


class MapDatasetEventSampler(MapDatasetEventSampler):
    def sample_sources(self, dataset):
        """Sample source model components.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Map dataset

        Returns
        -------
        events : `~gammapy.data.EventList`
            Event list
        """

        events_all = []
        for idx, evaluator in enumerate(dataset.evaluators.values()):
            if evaluator.needs_update:
                evaluator.update(
                    dataset.exposure,
                    dataset.psf,
                    dataset.edisp,
                    dataset._geom,
                    dataset.mask,
                )

            if evaluator.model.temporal_model is None:
                temporal_model = ConstantTemporalModel()
            else:
                temporal_model = evaluator.model.temporal_model

            try:
                if temporal_model.is_energy_dependent:
                    table = self._sample_coord_time_energy(dataset, evaluator.model)
                else:
                    flux = evaluator.compute_flux()
                    npred = evaluator.apply_exposure(flux)
                    table = self._sample_coord_time(npred, temporal_model, dataset.gti)

                if len(table) == 0:
                    mcid = table.Column(name="MC_ID", length=0, dtype=int)
                    table.add_column(mcid)

                table["MC_ID"] = idx + 1
                table.meta["MID{:05d}".format(idx + 1)] = idx + 1
                table.meta["MMN{:05d}".format(idx + 1)] = evaluator.model.name

                events_all.append(EventList(table))
            except:
                print(f'WARN: can not sample events for evaluator: {list(dataset.evaluators.keys())[idx]}')

        return EventList.from_stack(events_all)

    @staticmethod
    def event_list_meta(dataset, observation):
        """Event list meta info.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Map dataset
        observation : `~gammapy.data.Observation`
            In memory observation

        Returns
        -------
        meta : dict
            Meta dictionary
        """
        # See: https://gamma-astro-data-formats.readthedocs.io/en/latest/events/events.html#mandatory-header-keywords  # noqa: E501
        meta = {}

        meta["HDUCLAS1"] = "EVENTS"
        meta["EXTNAME"] = "EVENTS"
        meta[
            "HDUDOC"
        ] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
        meta["HDUVERS"] = "0.2"
        meta["HDUCLASS"] = "GADF"

        meta["OBS_ID"] = observation.obs_id

        meta["TSTART"] = (observation.tstart - dataset.gti.time_ref).to_value("s")
        meta["TSTOP"] = (observation.tstop - dataset.gti.time_ref).to_value("s")

        meta["ONTIME"] = observation.observation_time_duration.to("s").value
        meta["LIVETIME"] = observation.observation_live_time_duration.to("s").value
        meta["DEADC"] = 1 - observation.observation_dead_time_fraction

        if observation.pointing.mode == PointingMode.POINTING:
            fixed_icrs = observation.pointing.fixed_icrs
            meta["RA_PNT"] = fixed_icrs.ra.deg
            meta["DEC_PNT"] = fixed_icrs.dec.deg
        elif observation.pointing.mode == PointingMode.DRIFT:
            tref = Time(
                (observation.tstart.mjd + observation.tstop.mjd) / 2,
                format='mjd'
            )
            fixed_icrs = observation.pointing.get_icrs(tref)
            meta["RA_PNT"] = fixed_icrs.ra.deg
            meta["DEC_PNT"] = fixed_icrs.dec.deg
        else:
            raise ValueError(
                f'pointing mode {observation.pointing.mode} not supported, choices are "DRIFT" or "POINTING"'
            )

        meta["EQUINOX"] = "J2000"
        meta["RADECSYS"] = "icrs"

        meta["CREATOR"] = "Gammapy {}".format(gammapy.__version__)
        meta["EUNIT"] = "TeV"
        meta["EVTVER"] = ""

        meta["OBSERVER"] = "Gammapy user"
        meta["DSTYP1"] = "TIME"
        meta["DSUNI1"] = "s"
        meta["DSVAL1"] = "TABLE"
        meta["DSREF1"] = ":GTI"
        meta["DSTYP2"] = "ENERGY"
        meta["DSUNI2"] = "TeV"
        meta[
            "DSVAL2"
        ] = f'{dataset._geom.axes["energy"].edges.min().value}:{dataset._geom.axes["energy"].edges.max().value}'  # noqa: E501
        meta["DSTYP3"] = "POS(RA,DEC)     "

        offset_max = np.max(dataset._geom.width).to_value("deg")
        meta[
            "DSVAL3"
        ] = f"CIRCLE({fixed_icrs.ra.deg},{fixed_icrs.dec.deg},{offset_max})"  # noqa: E501
        meta["DSUNI3"] = "deg             "
        meta["NDSKEYS"] = " 3 "

        # get first non background model component
        for model in dataset.models:
            if model is not dataset.background_model:
                break
        else:
            model = None

        if model:
            meta["OBJECT"] = model.name
            meta["RA_OBJ"] = model.position.icrs.ra.deg
            meta["DEC_OBJ"] = model.position.icrs.dec.deg

        meta["TELAPSE"] = dataset.gti.time_sum.to("s").value
        meta["MJDREFI"] = int(dataset.gti.time_ref.mjd)
        meta["MJDREFF"] = dataset.gti.time_ref.mjd % 1
        meta["TIMEUNIT"] = "s"
        meta["TIMESYS"] = dataset.gti.time_ref.scale
        meta["TIMEREF"] = "LOCAL"
        meta["DATE-OBS"] = dataset.gti.time_start.isot[0][0:10]
        meta["DATE-END"] = dataset.gti.time_stop.isot[0][0:10]
        meta["CONV_DEP"] = 0
        meta["CONV_RA"] = 0
        meta["CONV_DEC"] = 0

        meta["NMCIDS"] = len(dataset.models)

        # Necessary for DataStore, but they should be ALT and AZ instead!
        telescope = observation.aeff.meta["TELESCOP"]
        instrument = observation.aeff.meta["INSTRUME"]
        if telescope == "CTA":
            if instrument == "Southern Array":
                loc = observatory_locations["cta_south"]
            elif instrument == "Northern Array":
                loc = observatory_locations["cta_north"]
            else:
                loc = observatory_locations["cta_south"]

        else:
            loc = observatory_locations[telescope.lower()]

        if observation.pointing.mode == PointingMode.POINTING:
            # this is not really correct but maybe OK for now
            coord_altaz = observation.pointing.get_altaz(dataset.gti.time_start, loc)

            meta["ALT_PNT"] = str(coord_altaz.alt.deg[0])
            meta["AZ_PNT"] = str(coord_altaz.az.deg[0])
        elif observation.pointing.mode == PointingMode.DRIFT:
            meta["ALT_PNT"] = observation.pointing.fixed_altaz.alt.deg
            meta["AZ_PNT"] = observation.pointing.fixed_altaz.az.deg
        else:
            raise ValueError(
                f'pointing mode {observation.pointing.mode} not supported, choices are "DRIFT" or "POINTING"'
            )

        # TO DO: these keywords should be taken from the IRF of the dataset
        meta["ORIGIN"] = "Gammapy"
        meta["TELESCOP"] = observation.aeff.meta["TELESCOP"]
        meta["INSTRUME"] = observation.aeff.meta["INSTRUME"]
        meta["N_TELS"] = ""
        meta["TELLIST"] = ""

        meta["CREATED"] = ""
        meta["OBS_MODE"] = observation.pointing.mode.name
        meta["EV_CLASS"] = ""

        return meta
