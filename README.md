# srcsim

## Description
Observation simulator for DL2-level IACT data, originally focused on CTA/LST telescope. Simulator uses the specified Monte Carlo simulation files to generate the mock event lists, corresponding to the indicated cosmic sources. Several observational setups are supported (the so-called "wobble", "on" modes), with the mock data optionally split into "observational runs" - similar to those of the real telescope. Cosmic ray sources may be simulated following the analytical spatial / spectral models (several optional are supported) or the FITS cube, defining the spectrum and morphology simultaneously.

## Installation
Clone and install with `pip`:

```
git clone https://gitlab.com/lageslise/source_simulator.git
cd source_simulator
pip install .
```

## Usage
There are two main executables to call:

* `getruns`: generates the observational run specifications following the specified configuration file.
* `simrun`: performs the simulation for the specified run configuration file.

The example of the configuration file for `getruns` may be found in the `examples` directory. When run as `getruns --config examples/sim_in_box.yaml` it will create configuration files for the separate runs in the `out` directory (needs to be created). This configuration files can be then processed with `simrun` one by one.

## Support
Please use [issues](https://gitlab.com/lageslise/source_simulator/issues) to report problems or make suggestions.

## Roadmap
Despite the initial focus on CTA/LST, the project may be extended to other instruments using the DL2 data format (e.g. other CTA instruments).

## Contributing
Contributions are welcome.

## Authors and acknowledgment
Original developers are Ievgen Vovk, Marcel Strzys and Elise Lagarde.

## License
We're using GNU GPLv3 here.

## Project status
Active development, so major changes are possible without a notice.
