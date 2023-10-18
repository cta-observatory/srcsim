import uproot
import pandas
import astropy.units as u
from dataclasses import dataclass


@dataclass(frozen=True)
class McInfo:
    num_showers: u.Quantity
    energy_range_min: u.Quantity
    energy_range_max: u.Quantity
    spectral_index: u.Quantity
    max_scatter_range: u.Quantity
    min_scatter_range: u.Quantity
    max_viewcone_radius: u.Quantity
    min_viewcone_radius: u.Quantity


@dataclass(frozen=True)
class MagicMcInfo(McInfo):
    file_name: str = ''

    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'File name':.<20s}: {self.file_name}
    {'Events':.<20s}: {self.num_showers}
    {'Energy min':.<20s}: {self.energy_range_min}
    {'Energy max':.<20s}: {self.energy_range_max}
    {'Spectral index':.<20s}: {self.spectral_index}
    {'Scatter range min':.<20s}: {self.min_scatter_range}
    {'Scatter range max':.<20s}: {self.max_scatter_range}
    {'Viewcone min':.<20s}: {self.min_viewcone_radius}
    {'Viewcone max':.<20s}: {self.max_viewcone_radius}
"""
        )

        return super().__repr__()

    @classmethod
    def from_file(cls, file_name):
        meta = dict()
        with uproot.open(file_name) as input_file:
            meta['num_showers'] = u.one * int(
                input_file['RunHeaders']['MMcRunHeader_1./MMcRunHeader_1.fNumSimulatedShowers'].array()[0]
            )
            meta['energy_range_min'] = u.GeV * input_file['RunHeaders']['MMcCorsikaRunHeader./MMcCorsikaRunHeader.fELowLim'].array()[0]
            meta['energy_range_max'] = u.GeV * input_file['RunHeaders']['MMcCorsikaRunHeader./MMcCorsikaRunHeader.fEUppLim'].array()[0]
            meta['spectral_index'] = u.one * input_file['RunHeaders']['MMcRunHeader_1.fSlopeSpec'].array()[0]
            meta['max_scatter_range'] = u.cm * input_file['RunHeaders']['MMcRunHeader_1.fImpactMax'].array()[0]
            meta['min_scatter_range'] = u.cm * 0
            meta['max_viewcone_radius'] = u.deg * input_file['RunHeaders']['MMcRunHeader_1./MMcRunHeader_1.fRandomPointingConeSemiAngle'].array()[0]
            meta['min_viewcone_radius'] = u.deg * 0

        info = MagicMcInfo(
            num_showers = meta['num_showers'],
            energy_range_min = meta['energy_range_min'],
            energy_range_max = meta['energy_range_max'],
            spectral_index = meta['spectral_index'],
            max_scatter_range = meta['max_scatter_range'],
            min_scatter_range = meta['min_scatter_range'],
            max_viewcone_radius = meta['max_viewcone_radius'],
            min_viewcone_radius = meta['min_viewcone_radius']
        )

        return info
    
    def to_df(self):
        units = dict(
            num_showers = u.one,
            energy_range_min = u.TeV,
            energy_range_max = u.TeV,
            spectral_index = u.one,
            max_scatter_range = u.m,
            min_scatter_range = u.m,
            max_viewcone_radius = u.deg,
            min_viewcone_radius = u.deg,
        )
        data = {
            key: [
                self.__getattribute__(key).to(units[key]).value
            ]
            for key in units
        }
        return pandas.DataFrame(data=data)
