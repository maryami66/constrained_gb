from setuptools import setup
from os import path


current_dir = path.abspath(path.dirname(__file__))


with open("README.md", "r") as fh:
    long_description = fh.read()

with open(path.join(current_dir, 'requirements.txt'), 'r') as f:
    install_requires = f.read().split('\n')

setup(
      name='constrained_gb',
      version='0.1',
      author='Maryam Bahrami',
      author_email='bahrami@uni-hildesheim.de',
      description='constrained optimization for gradient boosting models with non-decomposable constraints',
      long_description=long_description,
      long_description_content_type="text/markdown",
      #url='http://github.com/storborg/funniest',
      license='MIT',
      packages=['constrained_gb'],
      keywords='constrained optimization, gradient boosting',
      classifiers=[
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: MIT License",
              "Operating System :: OS Independent"
              ],
      python_requires='>=3.6'
      )
