from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name="openvs",
      version=1.0,
      description="OpenVS package for AI accelereted large scale virtual screening",
      long_description=long_description,
      author="Guangfeng Zhou",
      classifiers=[
          'Programming Language :: Python :: 3.7',
      ],
      packages=find_packages(exclude=['benchmarks','experiments']),
    )
