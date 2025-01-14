import os
import re
from collections import namedtuple
from typing import Union, Callable, Iterable
from functools import cached_property

import h5py
import numpy as np

Hists = namedtuple('hists', ['hists', 'midpts'])
columns_to_extract = ['time', 'index', 'usrpl01', 'x1', 'x2', 'x3', 'pl04', 'pl05', 'pl06', 'u1', 'dt']

class Frames:
    """
    Particle Frames contain multiple snapshot of particles, the number and memory of frames are very large
    We hereafter operate them with the file: we assume that the frames are all similiar with the same attribute columns
    """
    def __init__(self, h5file:str):
        self.h5 = h5file

    @cached_property
    def ds_names(self):
        """Get all dataset names from the h5 file"""
        with h5py.File(self.h5, 'r') as file:
            return list(file.keys())

    @cached_property 
    def ds_name0(self):
        """Get the first dataset name"""
        with h5py.File(self.h5, 'r') as file:
            for name, obj in file.items():
                if isinstance(obj, h5py.Dataset):
                    return name
        return None

    @cached_property
    def colnum(self):
        """Get number of columns in datasets"""
        with h5py.File(self.h5, 'r') as file:
            first_ds = file[self.ds_name0]
            return first_ds.shape[1]

    @cached_property
    def ds_namelen(self):
        """Get length of first dataset name"""
        return len(self.ds_name0) if self.ds_name0 else 0

    @cached_property
    def cols(self):
        """Get column names from first dataset"""
        with h5py.File(self.h5, 'r') as file:
            return file[self.ds_name0].attrs['columns'].tolist()
    
    @cached_property
    def ds_ranges(self):
        ds_names = self.get_full_ds_names()
        print(f"Start from {ds_names[0]} to {ds_names[-1]}.")

    def get_specific_ds_names(self, pattern):
        """Get dataset names matching pattern"""
        matches = [s for s in self.ds_names if re.match(pattern, s)]
        if matches:
            return matches
        print("NO SUCH DATASET NAMES")
        return []
    
    def get_full_ds_names(self):
        return self.get_specific_ds_names(r'\d+$')

    def get_index_ds_names(self):
        return self.get_specific_ds_names(r'\d+_index')

    def get_flag_ds_names(self):
        return self.get_specific_ds_names(r'\d+_flag')
    
    def get_destroy_name(self):
        des_names = self.get_specific_ds_names(r'destroy')
        if len(des_names) > 1:
            print("More than one destroy file found!")
        return des_names[0] if des_names else None
    
    def num2dsname(self, num: int):
        """Convert number to zero-padded dataset name"""
        return str(num).zfill(self.ds_namelen)

    def dsname2num(self, name:str):
        """Convert dataset name to index number"""
        try:
            return self.ds_names.index(name)
        except ValueError:
            return -1
    
    def col2num(self, name:str):
        """Convert column name to index"""
        try:
            return self.cols.index(name)
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

    def find_indices_sorted(self, lst:Iterable[Union[int,float]], target:Union[Union[int,float],Iterable[Union[int,float]]]):
        """
        Find the indices of the sorted target in a sorted list.
        """
        positions = np.searchsorted(lst, target)

        mask = (positions<len(lst)) & (lst[positions]==target)
            # Create mapping of target to position
        if isinstance(target, (int, float)):
            return positions[0] if mask[0] else -1
        else:
            result = np.array([pos if m else -1 
                    for pos, m in zip(positions, mask)])
            return result if np.any(result != -1) else -1


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
        def get_specific_hist(df, density:bool=True, logbins:bool=False, bins:int=50):

            pldata = df[:,sindex]

            if logbins:
                assert all(element >0 for element in pldata)
                bins = np.logspace(np.log10(np.min(pldata[pldata>0])), np.log10(np.max(pldata)), bins)
            else:
                bins = np.linspace(np.min(pldata), np.max(pldata), bins)

            hist, bins_edges = np.histogram(pldata, bins=bins, density=False)
            if density:
                hist = hist / len(pldata)
            midpoints = (bins_edges[:-1]+bins_edges[1:])/2
            return Hists(hist, midpoints)
        return get_specific_hist

    def get_ek_hist(self, df, bins:int=50) -> Hists:
        """
        Get particles histogram
        """
        return (self.get_general_hists('usrpl01'))(df,density=True, bins=bins)

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
            return hdf[self.get_destroy_name()][:]

    

# function to append data arrays to the first dataset
def append_to_dataset(h5_path, new_data):
    pfs = Frames(h5_path)
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