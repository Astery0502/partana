import os
import re
from collections import namedtuple
from typing import Union, List, Tuple, Callable, Iterable
from functools import cached_property

import h5py
import numpy as np
import pandas as pd
import pyvista as pv

Hists = namedtuple('hists', ['hists', 'midpts'])
columns_to_extract = ['time', 'index', 'usrpl01', 'x1', 'x2', 'x3', 'pl04', 'pl05', 'pl06', 'u1', 'dt']

def tail_sorted(filename: str) -> int:
    """
    A criteria sorting a file string list depending on their last number 

    >>> files = ['a_1.txt', 'a_3.txt', 'a_2.txt']
    >>> sorted(files, key=tail_sorted)
    ['a_1.txt', 'a_2.txt', 'a_3.txt']
    """
    pattern = r'_(\d+)\.\w+'
    match = re.search(pattern, filename)
    if match:
        num = int(match.groups()[0])
        return num
    raise ValueError("No pattern found in the filename") 

def mid_sorted(filename: str) -> int:
    pattern = r'\w+.*(\d+)_.*\w+.+'
    match = re.search(pattern, filename)
    if match:
        num = int(match.groups()[0]) 
        return num
    raise ValueError("No pattern found in the filename") 

# save all csv files into h5 file, with array and index separately stored into two datasets
def csvs2h5(csvfiles:Iterable[str], h5path:str):
    """
    Transform csv files to a H5 file (ONE CSV ONE DATASET)

    The column name is like:

    'time' 'dt' 'x1' 'x2' 'x3' 'u1' 'u2' 'u3' 
    'pl01' 'pl02' 'pl03': gyradius, pitch angle, v_perp
    'pl04' 'pl05' 'pl06' 'pl07': four a_//
    'pl08' 'pl09' 'pl10' 'pl11' 'pl12' 'pl13' 'pl14': seven v_drift
    'usrpl01' 'usrpl02' 'usrpl03': Ek, gca/fl, absb
    'ipe' 'iteration' 'index'

    Here we add extra datasets: index and flag for quick search of INDICES and GOVERNING FUNCTION (>0 the gca, <0 the full lorentz)
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

            if 'ensemble' in csvfile:
                # prepare dataset name and save data to the dataset
                ds_all = re.search(r'_(\d+).csv',csvfile).group(1)
                ds_index = ds_all+'_index'
                ds_flag = ds_all+'_flag'
                hdf.create_dataset(ds_all, data=dfi.to_numpy())
                hdf.create_dataset(ds_index, data=dfi['index'].to_numpy())
                hdf.create_dataset(ds_flag, data=dfi['usrpl02'].to_numpy())
                hdf[ds_all].attrs['columns'] = dfi.columns.to_list()
            elif 'destroy' in csvfile:
                ds_des = 'destroy'
                hdf.create_dataset(ds_des, data=dfi.to_numpy())
                hdf[ds_des].attrs['columns'] = dfi.columns.to_list()
            else:
                print(f"Unexpected file: {csvfile}.")

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

# save the compound data type of attrs and index
def csvs2h5_v2(csvfiles:Iterable[str], h5path:str):

    # once the file exists, no need to save, add instead recommended
    if os.path.exists(h5path):
        print(f"Already Existence of a H5 FILE: {h5path}.")
        return

    with h5py.File(h5path, 'w') as hdf:
        for csvfile in csvfiles:
            # read dataframe from csv file and format the column name
            dfi = pd.read_csv(csvfile)
            dfi.columns = dfi.columns.str.replace(' ', '', regex=False)

            # prepare for the compound data type and data of 2d array attrs and 1d array index
            row, col = dfi.shape
            compound_dtype = np.dtype([('attrs', np.float64, (col,)), ('index', np.float64)]) # here we identify the type in the sub array of 2d/1d->1d/scalar
            cdata = np.zeros(row, dtype=compound_dtype)
            cdata['attrs'] = dfi.to_numpy()
            cdata['index'] = dfi['index'].to_numpy()

            # prepare dataset name and save data to the dataset
            dataset_name = re.search(r'_(\d+).csv',csvfile).group(1)
            hdf.create_dataset(dataset_name, data=cdata)
            hdf[dataset_name].attrs['columns'] = [s.replace(" ","") for s in dfi.columns.to_list()]

def get_compound_dtype(df):
    """
    Create a compound dtype for the DataFrame.
    """
    dtype = []
    for column in df.columns:
        if pd.api.types.is_integer_dtype(df[column]):
            dtype.append((column, 'i4'))
        elif pd.api.types.is_float_dtype(df[column]):
            dtype.append((column, 'f4'))
        elif pd.api.types.is_string_dtype(df[column]):
            max_len = df[column].str.len().max()
            dtype.append((column, f'S{max_len}'))
        else:
            raise ValueError(f"Unsupported dtype for column {column}")
    return np.dtype(dtype)

def csvs2h5_structured(csvfiles:Iterable[str], h5path:str):
    if os.path.exists(h5path):
        print(f"Already Existence of a H5 FILE: {h5path}.")
        return
    with h5py.File(h5path, 'w') as hdf:
        for csvfile in csvfiles:
            dfi = pd.read_csv(csvfile)
            dfi.columns = dfi.columns.str.replace(' ', '', regex=False)
            dataset_name = re.search(r'_(\d+).csv',csvfile).group(1)
            compound_dtype = get_compound_dtype(dfi)
            structured_array = np.zeros(dfi.shape[0], dtype=compound_dtype)
            for col in dfi.columns:
                    structured_array[col] = dfi[col].values
            # Create dataset with compound data type
            dset = hdf.create_dataset(dataset_name, data=structured_array)
            # Add column names as attributes
            dset.attrs['columns'] = dfi.columns.tolist()

# single file to the h5 file
def csv2h5(csvfile:str, h5path:str):
    """
    'time' 'dt' 'x1' 'x2' 'x3' 'u1' 'u2' 'u3' 
    'pl01' 'pl02' 'pl03': gyradius, pitch angle, v_perp
    'pl04' 'pl05' 'pl06' 'pl07': four a_//
    'pl08' 'pl09' 'pl10' 'pl11' 'pl12' 'pl13' 'pl14': seven v_drift
    'usrpl01' 'usrpl02' 'usrpl03': Ek, gca/fl, absb
    'ipe' 'iteration' 'index'
    """
    if os.path.exists(h5path):
        print(f"Already Existence of a H5 FILE: {h5path}.")
        return
    with h5py.File(h5path, 'w') as hdf:
        dfi = pd.read_csv(csvfile)
        dataset_name = "des"# re.search(r'_(destroy).csv',csvfile).group()
        hdf.create_dataset(dataset_name, data=dfi.to_numpy())
        hdf[dataset_name].attrs['columns'] = [s.replace(" ","") for s in dfi.columns.to_list()]

# Extract vtk points to csv particle initial postions
def vtkpoints2csv(vtk_in:str, csv_out:str):
    mesh = pv.read(vtk_in)
    np.savetxt(csv_out, mesh.points, delimiter=' ')
    return mesh


class ParticleFrames:
    """
    Particle Frames contain multiple snapshot of particles, the number and memory of frames are very large
    We hereafter operate them with the file: we assume that the frames are all similiar with the same attribute columns
    """
    def __init__(self, h5file:str):
        self.h5 = h5file

    @cached_property
    def ds_names(self):
        """
        Get the all dataset names of the h5 file, note that the dataset names
        can not be indexed as the data is not compound
        """
        dataset_names = []

        def extractor(name, obj):
            if isinstance(obj, h5py.Dataset):
                dataset_names.append(name)

        with h5py.File(self.h5, 'r') as file:
            file.visititems(extractor)

        return dataset_names
    
    @cached_property
    def ds_name0(self):
        def find_first_ds(name, obj):
            if isinstance(obj, h5py.Dataset):
                return name        
        with h5py.File(self.h5, 'r') as hdf:
            first_ds_name = hdf.visititems(find_first_ds)
        return first_ds_name

    @cached_property
    def colnum(self):
        """
        Get the dataset shape from the datasets, assumed that all datasets
        have the same shape, which follows the format (part num, attr num)
        NOTE: REQUIRED TO CHECK ALL COLNUM EQUALS
        """
        with h5py.File(self.h5, 'r') as file:
            first_ds_name = self.ds_name0
            first_shape = file[first_ds_name].shape
        return first_shape[1]

    @cached_property
    def ds_namelen(self):
        first_ds_name = self.ds_name0
        return len(first_ds_name)

    @cached_property
    def cols(self):
        first_ds_name = self.ds_name0
        with h5py.File(self.h5, 'r') as hdf:
            ds_cols = hdf[first_ds_name].attrs['columns'].tolist()
        return ds_cols
    
    @cached_property
    def ds_ranges(self):
        ds_names = self.get_full_ds_names()
        print(f"Start from {ds_names[0]} to {ds_names[-1]}.")

    def get_specific_ds_names(self, pattern):
        ds_names = self.ds_names
        sds_names = [s for s in ds_names if re.match(pattern, s)]
        if len(sds_names) > 0:
            return sds_names
        else:
            print("NO SUCH DATASET NAMES")
    
    def get_full_ds_names(self):
        return self.get_specific_ds_names(r'\d+$')

    def get_index_ds_names(self):
        return self.get_specific_ds_names(r'\d+_index')

    def get_flag_ds_names(self):
        return self.get_specific_ds_names(r'\d+_flag')
    
    def get_destroy_name(self):
        des_name = self.get_specific_ds_names(r'destroy')
        if len(des_name) > 1:
            print("More than one destroy file found!")
        else:
            return des_name[0]
    
    def num2dsname(self, num: int):
        namelen = self.ds_namelen
        return str(num).zfill(namelen)

    def dsname2num(self, name:str):
        dataset_names = self.ds_names
        try:
            index = dataset_names.index(name)
            return index
        except ValueError:
            return -1
    
    def col2num(self, name:str):
        cols = self.cols
        try:
            index = cols.index(name)
            return index
        except ValueError:
            return -1

    @staticmethod
    def find_approx_index(lst:Iterable[Union[int,float]], target:Union[int,float]):
        """
        Find the index of the target in an approximately incrementing list.
        """
        indices =  np.where(lst == target)[0]
        if indices.size == 1:
            return indices[0]
        return -1
    
    @staticmethod
    def find_indices(lst:Iterable[Union[int,float]], target:Union[Union[int,float],Iterable[Union[int,float]]]):
        """
        find in the LIST the TARGET, return the boolean array or -1 if not found.
        """
        mask = np.isin(lst, target)
        indices = np.nonzero(mask)[0]
        if indices.size == 0:
            return -1
        return indices

    def general_extract(self, pl:str) -> Callable:
        """
        General Extract from one dataset where particles with one column property in selected range.
        """
        if isinstance(pl, str):
            sindex = self.col2num(pl)
        else:
            sindex = pl
        def extract(ds,v1:float,v2:float):
            assert v2>v1
            mask = (ds[:,sindex]<v2) & (ds[:,sindex]>v1)
            return ds[:][np.where(mask)[0]]
            # here we use the whole dataset to index as the mask indices can be very large leading to inefficient large amount of io read.
        return extract

    def extract_from_e(self, df, e1:float, e2:float):
        return self.general_extract('u3')(df, e1, e2)
    
    def extract_from_t(self, df, t1:float, t2:float):
        return self.general_extract('time')(df, t1, t2)

    def extract_from_ke(self, df, ke1:float, ke2:float):
        return self.general_extract('usrpl01')(df, ke1, ke2)

    def index2singpart(self, index: int):
        """
        Extract single particle infomation over time (datasets)
        """
        dataset_names = self.get_full_ds_names()
        arr = np.zeros((len(dataset_names), self.colnum))

        with h5py.File(self.h5, 'r') as hdf:
            for i,dataset_name in enumerate(dataset_names):
                ds = hdf[dataset_name]
                indices = hdf[dataset_name+'_index'][:]
                lindex = self.find_indices(indices, index)
                if lindex == -1:
                    break
                arr[i,:] = ds[lindex[0],:]
        arr = arr[~np.all(arr == 0, axis=1)]
        return arr
    
    def indices2frame(self, indices: Iterable[int]):
        """
        Extract multiple particle trajectory with index from all frames
        """
        dataset_names = self.get_full_ds_names()
        arr = []
        with h5py.File(self.h5, 'r') as hdf:
            for i,dataset_name in enumerate(dataset_names):
                ds = hdf[dataset_name]
                all_indices = hdf[dataset_name+'_index'][:]
                lindex = self.find_indices(all_indices, indices)
                if isinstance(lindex,int):
                    break
                arr.append(ds[lindex,:])
        return np.vstack(arr)

    def extract_frame(self, findex):
        with h5py.File(self.h5, 'r') as hdf:
            ds = hdf[self.num2dsname(findex)][:]
        return ds

    def get_general_hists(self, pl:Union[str,int]) -> Callable:
        # General Abstract get histogram from one one 2d array columns, the column is from the h5 column name 
        if isinstance(pl, str):
            sindex = self.col2num(pl)
        else:
            sindex = pl
        def get_specific_hist(df, density:bool=True, log:bool=True, bins:int=50):
            pldata = df[:,sindex]
            if log:
                assert all(element >0 for element in pldata)
                bins = np.logspace(np.log10(pldata.min()),np.log10(pldata.max()),bins)
            hist, bins_edges = np.histogram(pldata, bins=bins, density=False)
            if density:
                hist = hist / len(pldata)
            midpoints = (bins_edges[:-1]+bins_edges[1:])/2
            return Hists(hist, midpoints)
        return get_specific_hist

    def get_ek_hist(self, df, log:bool=True, bins:int=50) -> Hists:
        """
        Get particles histogram
        """
        return (self.get_general_hists('usrpl01'))(df,density=True, log=log, bins=bins)

    def extract_frames(self, extractor:Callable, start: int=0, end: int=1, interval: int=1, savpath:str="."):
        # create new h5 file and name it
        new_h5_name = os.path.join(savpath, "h5_int_"+str(start)+"_"+str(end)+"_"+str(interval)+".h5")
        if os.path.exists(new_h5_name):
            print(f"Already existing {new_h5_name}.")
            return

        with h5py.File(new_h5_name, 'w') as hdfw:
            cols = self.cols
            dset = hdfw.create_dataset('ds',shape=(0,0),maxshape=(None,None), dtype='float64')
            dset.attrs['columns'] = cols

        # read from h5_file datasets and load it into the new one
        ds_names = self.get_full_ds_names()
        with h5py.File(self.h5, 'r') as hdf:
            for ds_name in ds_names[start:end:interval]:
                dsi = extractor(hdf[ds_name])
                if dsi.shape[0]==0:
                    print(f"Reach the ParticleFrame with no data: {ds_name}")
                    break
                append_to_dataset(new_h5_name, dsi)
                print(f"complete extract {ds_name} to {new_h5_name}")
    
    def extract_destroy(self):
        """
        Extract DESTROY particle frame to a NUMPY array
        """
        with h5py.File(self.h5, 'r') as hdf:
            ds_des = hdf[self.get_destroy_name()][:]
        return ds_des

class ParticleFrame:
    def __init__(self, frame, cols:Iterable) -> None:
        assert frame.shape[1] == len(cols)
        self.frame = frame
        self.cols = cols

# function to append data arrays to the first dataset
def append_to_dataset(h5_path, new_data):
    pfs = ParticleFrames(h5_path)
    ds_names = pfs.ds_names
    with h5py.File(h5_path, 'a') as file:
        dset = file[ds_names[0]]
        current_shape = dset.shape
        num_new_rows, num_new_cols = new_data.shape

        # Determine the new shape after appending the new data
        max_rows = current_shape[0] + num_new_rows
        max_cols = max(current_shape[1], num_new_cols)
        
        # Resize the dataset to accommodate the new data
        dset.resize((max_rows, max_cols))
        
        # Write the new data to the end of the dataset
        dset[current_shape[0]:max_rows, :num_new_cols] = new_data
                    

def print_item_info(name, obj): 
    """
    Visitor function to print information about each item in the HDF5 file.
    
    Parameters:
    - name (str): The name of the item.
    - obj (h5py.Group or h5py.Dataset): The item itself.
    """
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}")
        print(f"  Shape: {obj.shape}")
        print(f"  Data type: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")

def extract_h5(h5_file, extractor):
    with h5py.File(h5_file, 'r') as f:
        # Use visititems to traverse the file structure
        f.visititems(extractor)

extract_h5_info = lambda h5_file: extract_h5(h5_file, print_item_info)