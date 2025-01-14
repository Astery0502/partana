"""
partana: A Python package for handling particle data.

This package provides tools for working with particle data, particularly focused on
particle data handling and visualization.
"""

__version__ = "0.1.0"
__author__ = "Hao Wu"
__license__ = "MIT" 

# Version information tuple
VERSION_INFO = tuple(map(int, __version__.split(".")))

# Expose main functionality at package level
from .frames.frames import Frames
from .io.converter import csvs2h5, vtkpos2csv, points2lines, vtk2lines

# Define what should be available in "from simesh import *"
__all__ = [
    'Frames',
    'csv2h5',
    'vtkpos2csv',
    'points2lines',
    'vtk2lines',
]