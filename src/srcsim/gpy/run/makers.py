import numpy as np
import astropy.units as u

from astropy.time import Time

from gammapy.makers import MapDatasetMaker
from gammapy.maps import Map, MapAxis
from gammapy.makers.utils import (
    make_map_background_irf,
    make_map_exposure_true_energy,
)
from gammapy.data import PointingMode


class MapDatasetMaker(MapDatasetMaker):
    @staticmethod
    def make_exposure(geom, observation, use_region_center=True):
        """Make exposure map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference map geom.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        exposure : `~gammapy.maps.Map`
            Exposure map.
        """
        nsteps = 100
        
        if observation.pointing.mode == PointingMode.POINTING:
            if isinstance(observation.aeff, Map):
                return observation.aeff.interp_to_geom(
                    geom=geom,
                )
            return make_map_exposure_true_energy(
                pointing=observation.get_pointing_icrs(observation.tmid),
                livetime=observation.observation_live_time_duration,
                aeff=observation.aeff,
                geom=geom,
                use_region_center=use_region_center,
            )

        elif observation.pointing.mode == PointingMode.DRIFT:
            mjd_edges = np.linspace(observation.tstart.mjd, observation.tstop.mjd, num=nsteps)
            mjd = Time(
                (mjd_edges[1:] + mjd_edges[:-1]) / 2,
                format='mjd'
            )
            dmjd = (mjd_edges[1:] - mjd_edges[:-1])

            mjd = Time(mjd, format='mjd')
            dmjd = dmjd * u.d
            
            maps = [
                make_map_exposure_true_energy(
                    pointing=observation.get_pointing_icrs(tref),
                    livetime=livetime,
                    aeff=observation.aeff,
                    geom=geom,
                    use_region_center=use_region_center,
                )
                for tref, livetime in zip(mjd, dmjd)
            ]

            time_axis = MapAxis.from_bounds(
                0,
                len(maps),
                nbin=len(maps),
                node_type="center",
                name="time"
            )

            return Map.from_stack(maps, axis=time_axis).reduce('time', np.add)

        else:
            raise RuntimeError(
                f'exposure calculation for pointing mode "{observation.pointing.mode}" not implemeted yet'
            )

    def make_background(self, geom, observation):
        """Make background map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geom.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        background : `~gammapy.maps.Map`
            Background map.
        """
        nsteps = 100
        
        if observation.pointing.mode == PointingMode.POINTING:
            bkg = observation.bkg

            if isinstance(bkg, Map):
                return bkg.interp_to_geom(geom=geom, preserve_counts=True)
    
            use_region_center = getattr(self, "use_region_center", True)
    
            if self.background_interp_missing_data:
                bkg.interp_missing_data(axis_name="energy")
    
            if self.background_pad_offset and bkg.has_offset_axis:
                bkg = bkg.pad(1, mode="edge", axis_name="offset")
    
            return make_map_background_irf(
                pointing=observation.pointing,
                ontime=observation.observation_time_duration,
                bkg=bkg,
                geom=geom,
                oversampling=self.background_oversampling,
                use_region_center=use_region_center,
                obstime=observation.tmid,
            )

        elif observation.pointing.mode == PointingMode.DRIFT:
            use_region_center = getattr(self, "use_region_center", True)
            
            mjd_edges = np.linspace(observation.tstart.mjd, observation.tstop.mjd, num=nsteps)
            mjd = Time(
                (mjd_edges[1:] + mjd_edges[:-1]) / 2,
                format='mjd'
            )
            dmjd = (mjd_edges[1:] - mjd_edges[:-1])

            mjd = Time(mjd, format='mjd')
            dmjd = dmjd * u.d
            
            maps = [
                make_map_background_irf(
                    pointing=observation.get_pointing_icrs(tref),
                    ontime=livetime,
                    bkg=observation.bkg,
                    geom=geom,
                    oversampling=self.background_oversampling,
                    use_region_center=use_region_center,
                    obstime=tref,
                )
                for tref, livetime in zip(mjd, dmjd)
            ]

            time_axis = MapAxis.from_bounds(
                0,
                len(maps),
                nbin=len(maps),
                node_type="center",
                name="time"
            )

            return Map.from_stack(maps, axis=time_axis).reduce('time', np.add)
        else:
            raise RuntimeError(
                f'exposure calculation for pointing mode "{observation.pointing.mode}" not implemeted yet'
            )
