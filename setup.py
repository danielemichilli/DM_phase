#!/usr/bin/env python

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'dmphase'
DESCRIPTION = 'Find the best Dispersion Measure by using FFT phase angle.'
URL = 'https://github.com/danielemichilli/DM_phase'
EMAIL = 'danielemichilli@gmail.com'
AUTHOR = 'Andrew Seymour and Daniele Michilli'
REQUIRES_PYTHON = '==2.7.*'
VERSION = False

# What packages are required for this module to be executed?
REQUIRED = [
    'numpy>=1.13.3',
    'matplotlib>=2.1',
    'scipy>=0.19.1'
]

# ------------------------------------------------

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, '__version__.py')) as f:
        about['__version__'] = f.read()
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine')
        os.system('twine upload dist/*')

        self.status('Pushing git tags')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')
        
        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    py_modules=['DM_phase'],
    entry_points={
         'console_scripts': ['DM_phase=DM_phase:from_PSRCHIVE'],
     },
    install_requires=REQUIRED,
    include_package_data=True,
    license='GNU',
    classifiers=[
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Operating System :: Unix",
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
