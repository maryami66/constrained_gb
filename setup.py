from setuptools import setup
from os import path


current_dir = path.abspath(path.dirname(__file__))


with open("README.md", "r") as fh:
    long_description = fh.read()

with open(path.join(current_dir, 'requirements.txt'), 'r') as f:
    install_requires = f.read().split('\n')


setup(
    name='constrained_gb',
    version='0.0.4',
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
        "License :: OSI Approved :: GNU General Public License (GPLv3)",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6'
)
