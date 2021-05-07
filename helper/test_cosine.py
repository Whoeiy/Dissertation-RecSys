# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:41:08 2021

@author: JenniferYu
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
      name='cosine_helper',
      ext_modules=cythonize('cosine.pyx')
      )

