#  Copyright (c) 2020 Robert Bosch GmbH
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.


from setuptools import setup
from os import path


current_dir = path.abspath(path.dirname(__file__))


with open("README.md", "r") as fh:
    long_description = fh.read()

with open(path.join(current_dir, 'requirements.txt'), 'r') as f:
    install_requires = f.read().split('\n')


setup(
    name='constrained_gb',
    version='0.0.5',
    author='Maryam Bahrami',
    author_email='maryami_66@yahoo.com',
    packages=['constrained_gb'],
    url='https://github.com/maryami66/constrained_gb',
    license='GNU General Public License v3 or later (GPLv3+)',
    description='constrained optimization for gradient boosting models with non-decomposable constraints',
    long_description=long_description,
    long_description_content_type="text/markdown",
    setup_requires='numpy',
    install_requires=install_requires,
    keywords='constrained optimization, gradient boosting',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6'
)
