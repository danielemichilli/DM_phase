# DM_phase

Find the best Dispersion Measure for a pulse by maximising the coherent power accross the bandwidth. 
It is robust to complex burst structures and interference.

## Getting Started

The necessary dependences will be installed automatically, except `psrchive` (see Prerequisites).
Run `python DM_phase.py -h` for a list of arguments.
Basic example: `python DM_phase.py fname`, where `fname` is the filename of a PSRCHIVE object.
Alternatively, it is possible to run the function `get_DM` on a 2D numpy array representing the pulse waterfall.
See the function documentation for details.

### Prerequisites

Python module of `psrchive` is required to read PSRCHIVE files and it will not be installed automatically. See the [package page](http://psrchive.sourceforge.net/manuals/python/).

### Installing

```
pip install DM_phase
```

## Contributing

Merge requests will be considered.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/danielemichilli/DM_phase/tags). 

## Authors

* **Andrew Seymour** - *Algorithm definition and mathematical calculations* - [aseymourGBT](https://github.com/aseymourGBT)
* **Daniele Michilli** - *Main scripter* - [danielemichilli](https://github.com/danielemichilli)

See also the list of [contributors](https://github.com/danielemichilli/DM_phase/contributors) who participated in this project.

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details
