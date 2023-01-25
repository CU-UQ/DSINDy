#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='DSINDy',
      version='0.1',
      description='DSINDy Code',
      author='Jacqui Wentz',
      author_email='jacqueline.wentz@colorado.edu',
      packages=find_packages(include=['dsindy', 'dsindy.*']))
