import numpy
import uproot
import pandas
import astropy.units as u
from dataclasses import dataclass


@dataclass(frozen=True)
class EventSample:
    reco_src_x: u.Quantity
    reco_src_y: u.Quantity
    reco_alt: u.Quantity
    reco_az: u.Quantity
    reco_ra: u.Quantity
    reco_dec: u.Quantity
    reco_energy: u.Quantity
    src_x: u.Quantity
    src_y: u.Quantity
    ra_tel: u.Quantity
    dec_tel: u.Quantity
    az_tel: u.Quantity
    alt_tel: u.Quantity
    delta_t: u.Quantity
    gammaness: u.Quantity
    mjd: u.Quantity = u.Quantity([], unit='d')
    mc_alt: u.Quantity = u.Quantity([], unit='deg')
    mc_az: u.Quantity = u.Quantity([], unit='deg')
    mc_energy: u.Quantity = u.Quantity([], unit='TeV')
    mc_alt_tel: u.Quantity = u.Quantity([], unit='deg')
    mc_az_tel: u.Quantity = u.Quantity([], unit='deg')


@dataclass(frozen=True)
class McOrigEventSample:
    az_tel: u.Quantity
    alt_tel: u.Quantity
    energy: u.Quantity
    src_x: u.Quantity = u.Quantity([], unit='m')
    src_y: u.Quantity = u.Quantity([], unit='m')
    alt: u.Quantity = u.Quantity([], unit='deg')
    az: u.Quantity = u.Quantity([], unit='deg')


@dataclass(frozen=True)
class MagicStereoEvents(EventSample):
    file_name: str = ''

    def __str__(self) -> str:
        summary = \
f"""{type(self).__name__} instance
    {'File name':.<20s}: {self.file_name}
    {'MC':.<20s}: {self.is_mc}
    {'Events':.<20s}: {self.n_events}
"""
        return summary

    def __repr__(self):
        print(str(self))

        return super().__repr__()
    
    @property
    def n_events(self):
        return len(self.reco_alt)
    
    @property
    def is_mc(self):
        return len(self.mc_alt) > 0

    @classmethod
    def from_file(cls, file_name):
        cam2angle = 1.5 / 450 *u.Unit('deg/mm')
        north_offset = 7

        event_data = dict()

        data_array_list = [
            'MTriggerPattern_1.fPrescaled',
            'MRawEvtHeader_1.fStereoEvtNumber',
            'MRawEvtHeader_1.fDAQEvtNumber',
            'MRawEvtHeader_1.fTimeDiff',
            'MPointingPos_1.fZd',
            'MPointingPos_1.fAz',
            'MPointingPos_1.fRa',
            'MPointingPos_1.fDec',
            'MEnergyEst.fEnergy',
            'MHadronness.fHadronness',
            'MRawEvtHeader_1.fDAQEvtNumber',
            'MSrcPosCam_1.fX',
            'MSrcPosCam_1.fY',
            'MStereoParDisp.fDirectionX',
            'MStereoParDisp.fDirectionY',
            'MStereoParDisp.fDirectionRA',
            'MStereoParDisp.fDirectionDec',
            'MStereoParDisp.fDirectionAz',
            'MStereoParDisp.fDirectionZd',
        ]

        time_array_list = ['MTime_1.fMjd', 'MTime_1.fTime.fMilliSec', 'MTime_1.fNanoSec']

        mc_array_list = [
            'MMcEvt_1.fEnergy',
            'MMcEvt_1.fTheta',
            'MMcEvt_1.fPhi',
        ]

        names_mapping = {
            'MTriggerPattern_1.fPrescaled': 'trigger_pattern',
            'MRawEvtHeader_1.fStereoEvtNumber': 'stereo_event_number',
            'MRawEvtHeader_1.fDAQEvtNumber': 'daq_event_number',
            'MRawEvtHeader_1.fTimeDiff': 'delta_t',
            'MPointingPos_1.fZd': 'zd_tel',
            'MPointingPos_1.fAz': 'az_tel',
            'MPointingPos_1.fRa': 'ra_tel',
            'MPointingPos_1.fDec': 'dec_tel',
            'MSrcPosCam_1.fX': 'src_x',
            'MSrcPosCam_1.fY': 'src_y',
            'MStereoParDisp.fDirectionX': 'reco_src_x',
            'MStereoParDisp.fDirectionY': 'reco_src_y',
            'MStereoParDisp.fDirectionRA': 'reco_ra',
            'MStereoParDisp.fDirectionDec': 'reco_dec',
            'MStereoParDisp.fDirectionAz': 'reco_az',
            'MStereoParDisp.fDirectionZd': 'reco_zd',
            'MHadronness.fHadronness': 'hadronness',
            'MEnergyEst.fEnergy': 'reco_energy',
            'MMcEvt_1.fEnergy': 'mc_energy',
            'MMcEvt_1.fTheta': 'mc_zd',
            'MMcEvt_1.fPhi': 'mc_az'
        }
        
        data_units = {
            'delta_t': u.s,
            'src_x': u.mm,
            'src_y': u.mm,
            'reco_src_x': u.deg,
            'reco_src_y': u.deg,
            'reco_alt': u.deg,
            'reco_az': u.deg,
            'reco_zd': u.deg,
            'reco_ra': u.deg,
            'reco_dec': u.deg,
            'reco_energy': u.GeV,
            'gammaness': u.one,
            'hadronness': u.one,
            'mjd': u.d,
            'alt_tel': u.deg,
            'az_tel': u.deg,
            'zd_tel': u.deg,
            'ra_tel': u.deg,
            'dec_tel':u.deg,
            'mc_alt':u.deg,
            'mc_az':u.deg,
            'mc_zd':u.deg,
            'mc_energy': u.GeV,
        }

        with uproot.open(file_name) as input_file:
            if 'Events' in input_file:
                data = input_file['Events'].arrays(data_array_list)

                # Mapping the read structure to the alternative names
                for key in data.fields:
                    name = names_mapping[key]
                    event_data[name] = data[key]

                is_mc = 'MMcEvt_1.' in input_file['Events']
                if is_mc:
                    data = input_file['Events'].arrays(mc_array_list)

                    # Mapping the read structure to the alternative names
                    for key in data.fields:
                        name = names_mapping[key]
                        event_data[name] = data[key]

                    # Post processing
                    event_data['mc_zd'] = numpy.degrees(event_data['mc_zd'])
                    event_data['mc_az'] = numpy.degrees(event_data['mc_az'])
                    # Transformation from Monte Carlo to usual azimuth
                    event_data['mc_az'] = -1 * (event_data['mc_az'] - 180 + north_offset)
                else:
                    # Reading the event arrival time information
                    data = input_file['Events'].arrays(time_array_list)

                    # Computing the event arrival time
                    mjd = data['MTime_1.fMjd']
                    millisec = data['MTime_1.fTime.fMilliSec']
                    nanosec = data['MTime_1.fNanoSec']

                    event_data['mjd'] = mjd + (millisec / 1e3 + nanosec / 1e9) / 86400.0

            else:
                # The file is likely corrupted, so return empty arrays
                for key in names_mapping:
                    name = names_mapping[key]
                    event_data[name] = numpy.zeros(0)
                    
        event_data['mc_alt'] = 90 - event_data['mc_zd']
        event_data['alt_tel'] = 90 - event_data['zd_tel']
        event_data['reco_alt'] = 90 - event_data['reco_zd']
        event_data['gammaness'] = 1 - event_data['hadronness']
        
        for key in event_data:
            event_data[key] = event_data[key].to_numpy()
            
        for key in event_data:
            if key in data_units:
                event_data[key] = event_data[key] * data_units[key]

        event_data['reco_src_x'] /= cam2angle
        event_data['reco_src_y'] /= cam2angle

        is_mc = 'mc_energy' in event_data

        if is_mc:
            events = MagicStereoEvents(
                reco_src_x = event_data['reco_src_x'],
                reco_src_y = event_data['reco_src_y'],
                reco_alt = event_data['reco_alt'],
                reco_az = event_data['reco_az'],
                reco_ra = event_data['reco_ra'],
                reco_dec = event_data['reco_dec'],
                reco_energy = event_data['reco_energy'],
                src_x = event_data['src_x'],
                src_y = event_data['src_y'],
                ra_tel = event_data['ra_tel'],
                dec_tel = event_data['dec_tel'],
                az_tel = event_data['az_tel'],
                alt_tel = event_data['alt_tel'],
                delta_t = event_data['delta_t'],
                gammaness = event_data['gammaness'],
                mc_alt = event_data['mc_alt'],
                mc_az = event_data['mc_az'],
                mc_energy = event_data['mc_energy'],
                mc_alt_tel = event_data['alt_tel'],
                mc_az_tel = event_data['az_tel'],
                file_name = file_name
            )
        else:
            events = MagicStereoEvents(
                reco_src_x = event_data['reco_src_x'],
                reco_src_y = event_data['reco_src_y'],
                reco_alt = event_data['reco_alt'],
                reco_az = event_data['reco_az'],
                reco_ra = event_data['reco_ra'],
                reco_dec = event_data['reco_dec'],
                reco_energy = event_data['reco_energy'],
                src_x = event_data['src_x'],
                src_y = event_data['src_y'],
                ra_tel = event_data['ra_tel'],
                dec_tel = event_data['dec_tel'],
                az_tel = event_data['az_tel'],
                alt_tel = event_data['alt_tel'],
                delta_t = event_data['delta_t'],
                mjd = event_data['mjd'],
                gammaness = event_data['gammaness'],
                file_name = file_name
            )

        return events
    
    def to_df(self):
        units = dict(
            # delta_t = u.s,
            src_x = u.m,
            src_y = u.m,
            reco_src_x = u.m,
            reco_src_y = u.m,
            reco_alt = u.rad,
            reco_az = u.rad,
            reco_ra = u.rad,
            reco_dec = u.rad,
            reco_energy = u.TeV,
            gammaness = u.one,
            mjd = u.d,
            alt_tel = u.rad,
            az_tel = u.rad,
            ra_tel = u.rad,
            dec_tel =u.rad,
            mc_alt =u.rad,
            mc_az = u.rad,
            mc_energy = u.TeV,
            mc_alt_tel = u.rad,
            mc_az_tel = u.rad,
        )
        data = {
            key: self.__getattribute__(key).to(units[key]).value
            for key in units
            if len(self.__getattribute__(key))
        }
        return pandas.DataFrame(data=data)


@dataclass(frozen=True)
class MagicMcOrigEvents(McOrigEventSample):
    file_name: str = ''

    def __str__(self) -> str:
        summary = \
f"""{type(self).__name__} instance
    {'File name':.<20s}: {self.file_name}
    {'Events':.<20s}: {self.n_events}
"""
        return summary

    def __repr__(self):
        print(str(self))

        return super().__repr__()

    @property
    def n_events(self):
        return len(self.az_tel)

    @classmethod
    def from_file(cls, file_name):
        north_offset = 7 * u.deg

        data_array_list = [
            'MMcEvtBasic_1.fEnergy',
            'MMcEvtBasic_1.fTelescopePhi',
            'MMcEvtBasic_1.fTelescopeTheta',
            'MSrcPosCam_1.fX',
            'MSrcPosCam_1.fY',
        ]

        data_units = {
            'MMcEvtBasic_1.fEnergy': u.GeV,
            'MSrcPosCam_1.fX': u.mm,
            'MSrcPosCam_1.fY': u.mm,
            'MMcEvtBasic_1.fTelescopePhi': u.rad,
            'MMcEvtBasic_1.fTelescopeTheta': u.rad,
        }

        with uproot.open(file_name) as input_file:
            if 'OriginalMC' in input_file:
                data = input_file['OriginalMC'].arrays(data_array_list)
            else:
                # The file is likely corrupted, so return empty arrays
                data = {
                    key: numpy.zeros(0)
                    for key in data_array_list
                }

        event_data = {
            key: data[key].to_numpy() * data_units[key]
            for key in data.fields
        }

        event_data['MMcEvtBasic_1.fTelescopeAlt'] = numpy.pi/2 * u.rad - event_data['MMcEvtBasic_1.fTelescopeTheta']
        # Transformation from Monte Carlo to usual azimuth
        data['MMcEvtBasic_1.fTelescopePhi'] = -1 * (
            data['MMcEvtBasic_1.fTelescopePhi'] - (180 * u.deg + north_offset).to(data_units['MMcEvtBasic_1.fTelescopePhi'])
        )

        events = MagicMcOrigEvents(
            az_tel = event_data['MMcEvtBasic_1.fTelescopePhi'],
            alt_tel = event_data['MMcEvtBasic_1.fTelescopeAlt'],
            energy = event_data['MMcEvtBasic_1.fEnergy'],
            src_x = event_data['MSrcPosCam_1.fX'],
            src_y = event_data['MSrcPosCam_1.fY'],
            file_name = file_name
        )

        return events