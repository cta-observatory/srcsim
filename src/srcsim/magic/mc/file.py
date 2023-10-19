import numpy
import healpy
from dataclasses import dataclass

from .info import MagicMcInfo
from .events import MagicStereoEvents, MagicMcOrigEvents


@dataclass(frozen=True)
class MagicMcFile:
    file_name: str
    meta: MagicMcInfo
    events_triggered: MagicStereoEvents
    events_simulated: MagicMcOrigEvents

    def __str__(self) -> str:
        summary = \
f"""{type(self).__name__} instance
    {'File name':.<20s}: {self.file_name}
    {'Simulated events':.<20s}: {self.events_simulated.n_events}
    {'Triggered events':.<20s}: {self.events_triggered.n_events}
    {'Energy range':.<20s}: {self.meta.energy_range_min} - {self.meta.energy_range_max}
    {'Spectral index':.<20s}: {self.meta.spectral_index}
    {'Scatter range':.<20s}: {self.meta.min_scatter_range} - {self.meta.max_scatter_range}
    {'Viewcone':.<20s}: {self.meta.min_viewcone_radius} - {self.meta.max_viewcone_radius}
    {'Telescope azimuth':.<20s}: {self.events_simulated.az_tel.to('deg').min()} - {self.events_simulated.az_tel.to('deg').max()}
    {'Telescope altitude':.<20s}: {self.events_simulated.alt_tel.to('deg').min()} - {self.events_simulated.alt_tel.to('deg').max()}
"""
        return summary

    @classmethod
    def from_file(cls, file_name):
        self = MagicMcFile(
            file_name = file_name,
            meta = MagicMcInfo.from_file(file_name),
            events_triggered = MagicStereoEvents.from_file(file_name),
            events_simulated = MagicMcOrigEvents.from_file(file_name)
        )
        return self

    def write(self, file_name):
        self.meta.to_df().to_hdf(
            file_name,
            key='/simulation/run_config',
            mode='w'
        )
        self.events_triggered.to_df().to_hdf(
            file_name,
            key='/dl2/event/telescope/parameters/MAGIC_MAGICCam',
            mode='a'
        )

    def healpy_split(self, nside):
        npix = healpy.nside2npix(nside)

        pix_ids_triggered = healpy.ang2pix(
            nside,
            numpy.pi/2 - self.events_triggered.alt_tel.to('rad').value,
            self.events_triggered.az_tel.to('rad').value,
        )

        pix_ids_simulated = healpy.ang2pix(
            nside,
            numpy.pi/2 - self.events_simulated.alt_tel.to('rad').value,
            self.events_simulated.az_tel.to('rad').value,
        )

        files = []

        for pix_id in range(npix):
            n_simulated = u.one * numpy.sum(pix_ids_simulated == pix_id)
            meta = MagicMcInfo(
                num_showers = n_simulated,
                energy_range_min = self.meta.energy_range_min,
                energy_range_max = self.meta.energy_range_max,
                spectral_index = self.meta.spectral_index,
                max_scatter_range = self.meta.max_scatter_range,
                min_scatter_range = self.meta.min_scatter_range,
                max_viewcone_radius = self.meta.max_viewcone_radius,
                min_viewcone_radius = self.meta.min_viewcone_radius
            )

            data = self.events_simulated
            selection = pix_ids_simulated == pix_id
            events_simulated = MagicMcOrigEvents(
                az_tel = data.az_tel[selection],
                alt_tel = data.alt_tel[selection],
                energy = data.energy[selection],
                src_x = data.src_x[selection],
                src_y = data.src_y[selection],
                file_name = data.file_name
            )

            data = self.events_triggered
            selection = pix_ids_triggered == pix_id
            events_triggered = MagicStereoEvents(
                reco_alt = data.reco_alt[selection],
                reco_az = data.reco_az[selection],
                reco_ra = data.reco_ra[selection],
                reco_dec = data.reco_dec[selection],
                reco_energy = data.reco_energy[selection],
                src_x = data.src_x[selection],
                src_y = data.src_y[selection],
                ra_tel = data.ra_tel[selection],
                dec_tel = data.dec_tel[selection],
                az_tel = data.az_tel[selection],
                alt_tel = data.alt_tel[selection],
                delta_t = data.delta_t[selection],
                gammaness = data.gammaness[selection],
                mjd = data.mjd[selection] if len(data.mjd) else data.mjd,
                mc_alt = data.mc_alt[selection] if len(data.mc_alt) else data.mc_alt,
                mc_az = data.mc_az[selection] if len(data.mc_az) else data.mc_az,
                mc_energy = data.mc_energy[selection] if len(data.mc_energy) else data.mc_energy,
                file_name = self.events_triggered.file_name
            )

            files.append(
                MagicMcFile(
                    file_name = self.file_name,
                    meta = meta,
                    events_triggered = events_triggered,
                    events_simulated = events_simulated,
                )
            )

        return files