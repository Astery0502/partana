import os
import re
import h5py
import numpy as np
import pandas as pd
from typing import Iterable

import pyvista as pv
from scipy.interpolate import splprep, splev


def csvs2h5(csvfiles:Iterable[str], h5path:str):
    """
    Transform csv files to a H5 file 

    Each csv file is a time frame of all active particles, corresponding to a dataset

    The column name is like:

    'time' 'dt' 'x1' 'x2' 'x3' 'u1' 'u2' 'u3' 
    'pl01' 'pl02' 'pl03': gyradius, pitch angle, v_perp
    'pl04' 'pl05' 'pl06' 'pl07': four a_//
    'pl08' 'pl09' 'pl10' 'pl11' 'pl12' 'pl13' 'pl14': seven v_drift
    'usrpl01' 'usrpl02' 'usrpl03': Ek, gca/fl, absb
    'ipe' 'iteration' 'index'

    Add two extra datasets. One is the sorted indices, the other is the positions of them in original sequence
    """
    # once the file exists, no need to save, add instead recommended
    if os.path.exists(h5path):
        print(f"Already Existence of a H5 FILE: {h5path}.")
        return

    with h5py.File(h5path, 'w') as hdf:
        for csvfile in csvfiles:
            # read dataframe from csv file and format the column name
            dfi = pd.read_csv(csvfile)
            dfi.columns = dfi.columns.str.replace(' ', '', regex=False)
            dfi = dfi.sort_values('index').reset_index(drop=True)

            if 'ensemble' in csvfile:
                # prepare dataset name and save data to the dataset
                ds_all = re.search(r'_(\d+).csv',csvfile).group(1)
                ds_index = ds_all+'_index'

                # Create sorted datasets with optimized settings
                hdf.create_dataset(ds_all, data=dfi.to_numpy(), 
                                 chunks=True,  # Enable chunking
                                 compression='gzip',  # Use gzip compression
                                 compression_opts=4,  # Moderate compression level
                                 fletcher32=True)    # Enable checksum for data integrity
                
                # Create sorted index dataset with optimization for faster searching
                hdf.create_dataset(ds_index, data=dfi['index'].to_numpy(),
                                 chunks=True,
                                 compression='gzip',
                                 compression_opts=4,
                                 fletcher32=True)
                
                # Store metadata
                hdf[ds_all].attrs['columns'] = dfi.columns.to_list()
                hdf[ds_all].attrs['shape'] = dfi.shape
            elif 'destroy' in csvfile:
                ds_des = 'destroy'
                hdf.create_dataset(ds_des, data=dfi.to_numpy())
                hdf[ds_des].attrs['columns'] = dfi.columns.to_list()
            else:
                print(f"Unexpected amrvac output file: {csvfile}.")

# Combine all csv files into one hdf5 file; unstructured version
def csvs2h5_v1(csvfiles:Iterable[str], h5path:str):
    """
    """
    if os.path.exists(h5path):
        print(f"Already Existence of a H5 FILE: {h5path}.")
        return
    with h5py.File(h5path, 'w') as hdf:
        for csvfile in csvfiles:
            dfi = pd.read_csv(csvfile)
            dataset_name = re.search(r'_(\d+).csv',csvfile).group(1)
            hdf.create_dataset(dataset_name, data=dfi.to_numpy())
            hdf[dataset_name].attrs['columns'] = [s.replace(" ","") for s in dfi.columns.to_list()]

# Extract vtk points to csv particle initial postions
def vtkpos2csv(vtk_in:str, csv_out:str):
    mesh = pv.read(vtk_in)
    np.savetxt(csv_out, mesh.points, delimiter=' ')
    return mesh

def points2lines(points, point_data, sample_points=0):

    """ transform points to line-like points array """

    # Check if point_data contains integers
    if np.issubdtype(point_data.dtype, np.integer):
        # Find indices where values change and are non-zero
        changes = np.where((point_data[1:] != point_data[:-1]) & (point_data[1:] != 0))[0] + 1
        start_indices = changes
    else:
        # Find indices where values are 0
        start_indices = np.where(point_data == 0)[0]

    lines = []
    for i, start_idx in enumerate(start_indices):
        # Get end index for current line segment
        end_idx = start_indices[i+1] if i < len(start_indices)-1 else None
        line = points[start_idx:end_idx]

        if sample_points > len(line) and sample_points > 0:
            # Use spline interpolation to get more points
            tck, _ = splprep([line[:,j] for j in range(3)], s=0)
            lines.append(np.array(splev(np.linspace(0, 1, sample_points), tck)))
        else:
            lines.append(line.T)

    return lines

def vtk2lines(vtk, sample_points:int = 0):

    vtkpoints = pv.read(vtk)
    points = vtkpoints.points
    point_data = vtkpoints.point_data['colorVar']

    return points2lines(points, point_data, sample_points)