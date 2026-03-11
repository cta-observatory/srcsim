import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord


from .sample import MCSampleBase


def power_law(e, e0, norm, index):
    return norm * (e/e0).decompose()**index


class MCSample(MCSampleBase):
    def __init__(self, file_name=None, obs_id=None, data_table=None, config_table=None):
        self.units = dict(
            energy = u.TeV,
            angle = u.rad,
            distance = u.m,
            viewcone = u.deg
        )
        
        # TODO: refine the logic below / implement nicer
        if data_table is not None and config_table is not None:
            self.file_name = None
            self.obs_id = config_table['obs_id'].iloc[0]
            self.config_table = config_table
            self.data_table = data_table
        else:
            self.file_name = file_name
            self.obs_id = obs_id
            self.config_table = self.read_config(file_name, obs_id)
            self.data_table = self.read_data(file_name, obs_id)

        self.data_table = self._fix_format(self.data_table)
        self.config_table = self._fix_format(self.config_table)

        # Getting the telescope pointing
        pointing_data = self.data_table[['pointing_az', 'pointing_alt']].mean()
        self.tel_pos = SkyCoord(pointing_data['pointing_az'], pointing_data['pointing_alt'], unit=self.units['angle'], frame='altaz')
        
        # Working out the simulation spectrum
        rmin, rmax = self.config_table[['min_scatter_range', 'max_scatter_range']].iloc[0].values * self.units['distance']
        ground_area = np.pi * (rmax**2 - rmin**2)
        nevents = self.config_table['n_showers'].iloc[0] * self.config_table['shower_reuse'].iloc[0]
        emin = self.config_table['energy_range_min'].iloc[0] * self.units['energy']
        emax = self.config_table['energy_range_max'].iloc[0] * self.units['energy']
        index = self.config_table['spectral_index'].iloc[0]
        self.spec_data = self.get_spec_data(nevents, emin, emax, index)
        self.spec_data['norm'] /= ground_area

        alt, az = self.data_table[['true_alt', 'true_az']].to_numpy().transpose()
        _unit = 'rad' if 'mc_alt' in self.data_table.columns else 'deg'
        self.evt_coord = SkyCoord(az, alt, frame='altaz', unit=_unit)
        self.evt_coord = self.evt_coord.transform_to(self.tel_pos.skyoffset_frame())

        self.evt_energy = self.data_table['true_energy'].to_numpy() * self.units['energy']

        # Filtering out events with excessive offsets (e.g. due to the simulation numerical accuracy)
        offset_min, offset_max = self.config_table[['min_viewcone_radius', 'max_viewcone_radius']].iloc[0].values * self.units['viewcone']
        evt_offset = self.evt_coord.separation(self.tel_pos)

        in_fov = (evt_offset >= offset_min) & (evt_offset <= offset_max)
        self.data_table = self.data_table[in_fov]
        self.evt_coord = self.evt_coord[in_fov]
        self.evt_energy = self.evt_energy[in_fov]
        
    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'File name':.<20s}: {self.file_name}
    {'Obs ID':.<20s}: {self.obs_id}
    {'Pointing':.<20s}: {self.tel_pos}
    {'N events':.<20s}: {len(self.data_table)}
    {'Energy range':.<20s}: [{self.config_table['energy_range_min'].iloc[0]:.2e}; {self.config_table['energy_range_max'].iloc[0]:.2e}] {self.units['energy']}
    {'Viewcone':.<20s}: [{self.config_table['min_viewcone_radius'].iloc[0]:.1f}; {self.config_table['max_viewcone_radius'].iloc[0]:.1f}] {self.units['viewcone']}
"""
        )

        return super().__repr__()

    @classmethod
    def read_config(cls, file_name, obs_id):
        config = super().read_config(file_name)
        return config.query(f'obs_id == {obs_id}')
        
    @classmethod
    def read_data(cls, file_name, obs_id):
        data = super().read_data(file_name)
        return data.query(f'obs_id == {obs_id}')

    def _fix_format(self, table):
        table = table.copy()
        old2new = dict(
            mc_energy = 'true_energy',
            mc_az = 'true_az',
            mc_alt = 'true_alt',
            mc_az_tel = 'pointing_az',
            mc_alt_tel = 'pointing_alt',
            num_showers = 'n_showers'
        )
        for key in old2new:
            if key in table.columns:
                table[old2new[key]] = table[key]

        return table

    def get_spec_data(self, n_events, emin, emax, index=-1):
        e0 = (emin * emax)**0.5

        if index == -1:
            norm = n_events / e0 / (np.log(emax/e0) - np.log(emin/e0))
        else:
            norm = n_events * (index + 1) / e0 / ((emax/e0).decompose()**((index + 1)) - (emin/e0).decompose()**((index + 1)))

        norm = norm.to(1/u.eV)

        sim_spec = {
            'norm': norm,
            'e0': e0,
            'index': index
        }

        return sim_spec
    
    def dnde(self, energy):
        return power_law(energy, **self.spec_data)
    
    def dndo(self, coord):
        offset_min, offset_max = self.config_table[['min_viewcone_radius', 'max_viewcone_radius']].iloc[0].values * self.units['viewcone']
        sky_area = 2 * np.pi * (np.cos(offset_min) - np.cos(offset_max)) * u.sr
        norm = 1 / sky_area

        r = self.tel_pos.separation(coord)
        
        return norm * (r >= offset_min) * (r <= offset_max)

    def dndedo(self, energy, coord):
        return self.dnde(energy) * self.dndo(coord)
