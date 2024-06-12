import pandas as pd
import os
import yt
from dataclass import dataclass
from typing import Any, List, Union, Callable, Iterable


@dataclass
class subdomainArguments:
    dims: List[int]
    level: int
    left_edge: List[float]
    savpath: str

# load the AMRVAC data and transform it to b1,b2,b3 binary file for matlab ARD use, 
# see https://github.com/RainthunderWYL/LoRD/
def dat2bin(filepath:str, subargs:subdomainArguments) -> None:
    ds = yt.load(filepath)
    left_edge = subargs.left_edge
    dims = subargs.dims
    level = subargs.level
    savpath = subargs.savpath

    # Get the data for b1, b2 and b3 as Numpy arrays
    b1_data = ds.covering_grid(level, left_edge=left_edge, dims=dims*ds.refine_by**level)['b1'].in_units("code_magnetic").T
    b2_data = ds.covering_grid(level, left_edge=left_edge, dims=dims*ds.refine_by**level)['b2'].in_units("code_magnetic").T
    b3_data = ds.covering_grid(level, left_edge=left_edge, dims=dims*ds.refine_by**level)['b3'].in_units("code_magnetic").T

    b1_data.tofile(os.path.join(savpath, 'b1.bin')) 
    b2_data.tofile(os.path.join(savpath, 'b2.bin')) 
    b3_data.tofile(os.path.join(savpath, 'b3.bin')) 

def categorize_reconnection(ARDFile: str):
    # load the ARD file
    ARD = pd.read_csv(ARDFile)
    # categorize the reconnection events


