# DM_phase

Find the best Dispersion Measure for a pulse by maximising the coherent power accross the bandwidth. 
It is robust to complex burst structures and interference.

## Installation
The latest stable version is available on PyPI and can be installed with
```
pip install DM_phase
```

Alternatively, it is possible to clone the current repository and run
```
python setup.py install
```

### Prerequisites
The necessary dependences will be installed automatically, except for `psrchive` ([package page](http://psrchive.sourceforge.net/manuals/python/)).
The Python module of `psrchive` is only required to read PSRCHIVE files and can be ignored when using numpy arrays to load the burst.

## Usage
The script can be run from terminal on `psrchive` files.
Run
 `python DM_phase.py -h` for a list of arguments.
Basic example: `python DM_phase.py fname`, where `fname` is the filename of a PSRCHIVE object.

Alternatively, it is possible to run the function `get_DM` on a 2D numpy array representing the pulse waterfall.
See the function documentation for details.
A working example is provided under `docs/usage_example.py`.

An explanation of the diagnostic plots produced is given in `docs/What_am_I_looking_at.pdf`.
The mathematical background is outlined in `docs/mathematical_background.pdf`.

It is important to select a valid range of fluctuation frequencies. The software will try to select the best range. However, it may be needed to manually refine the range for very complex bursts. This can be done by passing the necessary values to the `get_DM` function (see `docs/usage_example.py`) or by using the graphical interface (`-manual_cutoff` option). Similar methods can also be used to select the burst is the waterfall.

## Contributing
Merge requests will be considered.

## Versioning
We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/danielemichilli/DM_phase/tags). 

## Authors
* **Andrew Seymour** - *Algorithm definition and mathematical calculations* - [aseymourGBT](https://github.com/aseymourGBT)
* **Daniele Michilli** - *Main scripter* - [danielemichilli](https://github.com/danielemichilli)
* **Ziggy Pleunis** - *Implementation of many features and code manteinance* [zpleunis](https://github.com/zpleunis)

See also the list of [contributors](https://github.com/danielemichilli/DM_phase/contributors) who participated in this project.

## License
This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details.

Please, cite ascl.net/1910.004 if you use this code in a publication
