import doctest
import re

import numpy as np
import pandas as pd
import dask.dataframe as dd 

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from collections import namedtuple
from typing import Union, List, Tuple, Callable, Iterable
from dask.distributed import Client
import dask.dataframe as dd

Hists = namedtuple('hists', ['hists', 'midpts'])
columns_to_extrat = ['time', 'index', 'usrpl01', 'x1', 'x2', 'x3', 'pl04', 'pl05', 'pl06', 'u1', 'dt']

def rebin_val(val: Iterable, bin:int) -> np.ndarray:
    val = np.array(val)
    padding_size = (bin - len(val) % bin) % bin
    padded_data = np.pad(val, (0, padding_size), 'constant')
    rebinned_val = padded_data.reshape(-1, bin).sum(axis=1)
    return rebinned_val

class ParticleFrame(dd.DataFrame):

    @property
    def _constructor(self):
        return ParticleFrame

    def tp_prep(self) -> None:
        """
        Preprocess the DataFrame: clean column names and (adjust angles).

        Example:
        >>> df = ParticleFrame({' x1': [1,2], ' x2': [3,4]})
        >>> df.columns.tolist()
        ['x1', 'x2']
        """
        # Clean up column names by removing spaces
        self.columns = [col.replace(" ","") for col in self.columns]

    def __init__(self, data, *args, **kwargs):
        if isinstance(data, str):
            data = dd.read_csv(data)
            data = data.repartition(npartitions=10)
        elif isinstance(data, pd.DataFrame):
            data = dd.from_pandas(data, npartitions=10)
        elif isinstance(data, dict):
            data = dd.from_pandas(pd.DataFrame(data), npartitions=10)
        elif isinstance(data, dd.DataFrame):
            data = data.repartition(npartitions=10)
        super().__init__(data._expr, *args, **kwargs)
        self.tp_prep()

    @staticmethod
    def get_val_range(val:str) -> Callable[[dd.DataFrame,float,float], dd.DataFrame]:
        def get_val_specified(df:dd.DataFrame, v1:float, v2:float):
            return df[(df[val]>=v1) & (df[val]<=v2)]
        return get_val_specified
    
    def get_trange(self, v1:float, v2:float) -> dd.DataFrame:
        """
        Get particles whose time ranges from v1 to v2        

        Example:
        >>> df = ParticleFrame({'time':[1,2,3], 'value':[2,3,5]})
        >>> (df.get_trange(1,2))['time'].compute().tolist()
        [1, 2]
        """
        return self.get_val_range('time')(self, v1, v2)

    def get_tuntil(self, v2:float) -> dd.DataFrame:
        """
        Get particles whose time ranges until v2
        """
        return self.get_trange(0, v2)

    def get_tfrom(self, v1:float) -> dd.DataFrame:
        """
        Get particles whose time ranges from v1
        """
        return self.get_trange(v1, 10000)

    def get_erange(self, e0:float, e1:float, ev: bool=True) -> dd.DataFrame:
        """
        Get particles whose energy ranges from e0 to e1, eV unit by default

        Example:
        >>> df = ParticleFrame({'usrpl01':[5,7,11], 'value':[1,2,3]})
        >>> (df.get_erange(7,11,ev=False))['usrpl01'].compute().tolist()
        [7, 11]
        """
        if ev:
            e0 /= 511
            e1 /= 511
        return self.get_val_range('usrpl01')(self,e0,e1)

    def get_indexed(self, indices: Union[int, Iterable[int]]) -> dd.DataFrame:
        """
        Get particles whose indices from the indices list or single int

        Example:
        >>> df = ParticleFrame({'index':[1,2,3], 'value':[1,2,3]})
        >>> (df.get_indexed([1,5]))['index'].compute().tolist()
        [1]
        >>> (df.get_indexed(1))['index'].compute().tolist()
        [1]
        """
        if isinstance(indices, int):
            return self[self['index'] == indices]
        return self.loc[self['index'].isin(indices)]
    
    def write_csv(self, path:str) -> None:
        return self.to_csv(path, sep=',', index=False, single_file=True)
    
    def get_right(self) -> dd.DataFrame:
        """
        Get particles located in the positive x-axis

        Example:
        >>> df = ParticleFrame({'x1':[-1,-2,3], 'value':[1,2,3]})
        >>> (df.get_right())['x1'].compute().tolist()
        [3]
        """
        return self.get_val_range('x1')(self,0,10000)
    
    def get_left(self) -> dd.DataFrame:
        """
        Get particles located in the negative x-axis

        Example:
        >>> df = ParticleFrame({'x1':[-1,-2,3], 'value':[1,2,3]})
        >>> (df.get_left())['x1'].compute().tolist()
        [-1, -2]
        """
        return self.get_val_range('x1')(self,-10000,0)
    
    @staticmethod
    def get_general_hists(pl:str) -> Callable:
        def get_specific_hist(df:dd.DataFrame, density:bool=True, log:bool=True, bins:int=True):
            pldata = df[pl].compute()
            if log:
                assert all(element >0 for element in pldata)
                bins = np.logspace(np.log10(pldata.min()),np.log10(pldata.max()),bins)
            hist, bins_edges = np.histogram(pldata, bins=bins, density=False)
            if density:
                hist = hist / len(pldata)
            midpoints = (bins_edges[:-1]+bins_edges[1:])/2
            return Hists(hist, midpoints)
        return get_specific_hist
    
    def get_ek_hist(self, hdf:Callable[[dd.DataFrame],dd.DataFrame]=lambda df:df, log:bool=True, bins:int=200) -> Hists:
        """
        Get particles histogram

        Example:
        >>> df = ParticleFrame({'usrpl01':[5,7,11], 'value':[1,2,3]})
        >>> df.get_ek_hist(bins=3, log=False)
        hists(hists=array([0.33333333, 0.33333333, 0.33333333]), midpts=array([ 6.,  8., 10.]))
        """
        return (self.get_general_hists('usrpl01'))(hdf(self),density=True, log=log, bins=bins)

class DesPartFrame(ParticleFrame):
    @property
    def _constructor(self):
        return DesPartFrame

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['hover_text'] = self.apply(lambda row: f"<b>Time:</b> {row['time']}<br>"
                        f"<b>Index:</b> {row['index']}<br>"
                        f"<b>Energy:</b> {row['usrpl01']}<br>"
                        f"<b>X:</b> {row['x1']}<br>"
                        f"<b>Y:</b> {row['x2']}<br>",
                        axis=1, meta=('hover_text', 'category'))
    
    def plot_bottom(self, hdf:Callable[[dd.DataFrame],dd.DataFrame]=lambda df:df) -> None:

        df = hdf(self)
        df_subset = df[columns_to_extrat+['hover_text']].compute()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=False, 
                vertical_spacing=0.1, subplot_titles=('Scatter Plot', 'Histogram'))
        fig.add_trace(go.Scatter(
            x=df_subset['x1'],
            y=df_subset['x2'],
            mode='markers',
            marker=dict(
                color=df_subset['usrpl01']*511,
                colorscale='Jet',
                colorbar=dict(title='Ek'),
            ),
            text=df_subset['hover_text'],
            hoverinfo='text',
            hovertemplate='%{text}<extra></extra>',
        ), row=1, col=1)

        ek_hists = df.get_ek_hist(bins=20)
        fig.add_trace(go.Scatter(
            x=511*ek_hists.midpts,
            y=ek_hists.hists,
            mode='markers',
            marker=dict(color='blue')
        ), row=2, col=1)

        fig.update_xaxes(type="log", row=2, col=1)
        fig.update_yaxes(type="log", row=2, col=1)

        fig.update_layout(
            coloraxis_colorbar=dict(
            x=0.95, y=0.5, len=0.75, thickness=20
            ),
            legend=dict(
            x=1.35, y=1, xanchor='right', yanchor='top'
            ),
            title='Bottom Electron Map',
            xaxis_title='X Axes',
            yaxis_title='Y Axis',
        )

        # Update layout with buttons
        fig.update_layout(
            width=1000,
            height=800,
            updatemenus=[
            dict(
            type="buttons",
            direction="left",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.1, xanchor="left",
            y=1.2, yanchor="top"
            )
            ])

        # Update individual subplot titles and axes labels
        fig.update_xaxes(title_text='X Axis', row=1, col=1)
        fig.update_yaxes(title_text='Y Axis', row=1, col=1)
        fig.update_xaxes(title_text='Ek', row=2, col=1)
        fig.update_yaxes(title_text='Counts', row=2, col=1)
        fig.show()

class SingPartFrame(ParticleFrame):
    @property
    def _constructor(self):
        return SingPartFrame

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot_traj(self, bin: int=1):
        # Add a hover text column to the DataFrame for the plot: time; index; energy; x1; x2; x3
        self['hover_text'] = self.apply(lambda row: f"<b>Time:</b> {row['time']}<br>"
                    f"<b>Index:</b> {row['index']}<br>"
                    f"<b>Energy:</b> {row['usrpl01']}<br>"
                    f"<b>X:</b> {row['x1']}<br>"
                    f"<b>Y:</b> {row['x2']}<br>"
                    f"<b>Z:</b> {row['x3']}<br>",
                    axis=1, meta=('hover_text', 'category'))
        #index = self['index'][0].compute()
        df_subset = self[columns_to_extrat+['hover_text']].compute()
        time = df_subset['time'][::bin]

        fig = make_subplots(
            rows = 1, cols = 2,
            specs=[[{'type': 'scatter3d'},
                   {'type': 'scatter'}]],
            subplot_titles=(f'Trajectory of Particle'),

        )

        fig.add_trace(go.Scatter3d(
            x = df_subset['x1'][::bin],
            y = df_subset['x2'][::bin],
            z = df_subset['x3'][::bin],
            mode='markers',
            marker=dict(
                size=5,
                color=time,
                colorscale='Jet',
                opacity=0.9
            ),
            text=df_subset['hover_text'][::bin],
            hoverinfo='text',
            hovertemplate='%{text}<extra></extra>',
            name='Trajectory'
        ), row=1, col=1)

        # calculate dek values
        df_subset['dek1'] = df_subset['pl04'] * df_subset['u1'] * df_subset['dt']
        df_subset['dek2'] = df_subset['pl05'] * df_subset['u1'] * df_subset['dt']
        df_subset['dek3'] = df_subset['pl06'] * df_subset['u1'] * df_subset['dt']
        print("dek1:", df_subset['dek1'].sum())
        print("dek2:", df_subset['dek2'].sum())
        print("dek3:", df_subset['dek3'].sum())

        # Add the time series scatter plots
        for i, u in enumerate(['dek1', 'dek2', 'dek3'], start=1):
            dek = rebin_val(df_subset[u], bin)
            fig.add_trace(go.Scatter(x=time, y=dek, 
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=time,
                            colorscale='Jet',
                            opacity=0.9
                            ),
                        text=self['hover_text'][::bin],
                        hoverinfo='text',
                        hovertemplate='%{text}<extra></extra>',
                        name=f'Curve{i}'
                        ), row=1, col=2)

        fig.data[1].visible = True
        fig.data[2].visible = False
        fig.data[3].visible = False
        # Create buttons for the layout to toggle the visible trace
        fig.update_layout(
            width = 1500,
            height = 800,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=1.01,
                    xanchor="left",
                    y=0.8,
                    yanchor="top",
                    buttons=[
                        dict(label="ep",
                            method="update",
                            args=[{"visible": [True, True, False, False]},
                                {"title": "Time series of Ep"}]),
                        dict(label="gradb",
                            method="update",
                            args=[{"visible": [True, False, True, False]},
                                {"title": "Time series of Gradb"}]),
                        dict(label="curvb",
                            method="update",
                            args=[{"visible": [True, False, False, True]},
                                {"title": "Time series of Curvb"}]),
                    ],
                )
            ],
            # Adjust the domain to make the first subplot larger
            scene = dict(domain=dict(x=[0,0.50])),
            xaxis2 = dict(domain=[0.55,1.0]),
        ) 

        # Update axes titles
        fig.update_xaxes(title_text='Time', row=1, col=2)
        fig.update_yaxes(title_text='Value', row=1, col=2)

        fig.show()

class EnsembleFrames:

    def __init__(self, files: Iterable[str]):
        self.ensemble = files
    
    # criteria to sort file by the last number before .
    @staticmethod
    def tail_sorted(filename: str) -> int:
        """
        A criteria sorting a file string list depending on their last number 

        >>> files = ['a_1.txt', 'a_3.txt', 'a_2.txt']
        >>> tail_sorted = EnsembleFrames.tail_sorted
        >>> sorted(files, key=tail_sorted)
        ['a_1.txt', 'a_2.txt', 'a_3.txt']
        """
        pattern = r'_(\d+)\.\w+'
        match = re.search(pattern, filename)
        if match:
            num = int(match.groups()[0])
            return num
        raise ValueError("No pattern found in the filename") 

    @staticmethod
    def mid_sorted(filename: str) -> int:
        pattern = r'\w+.*(\d+)_.*\w+.+'
        match = re.search(pattern, filename)
        if match:
            num = int(match.groups()[0]) 
            return num
        raise ValueError("No pattern found in the filename") 
    
    def derive_single_traj(self, index: int) -> dd.DataFrame:
        """ 
        The self.ensemble is supposed to be sorted already.
        """
        df = ParticleFrame(self.ensemble[0])
        dfa = SingPartFrame(pd.DataFrame())
        for f in self.ensemble:
            df = ParticleFrame(f)
            dfi = df.get_indexed(index)
            if len(dfi.index)==0:
                print(f"No more data from the index: {index} in the file: {f}")
                break
            dfa = dd.concat([dfa.compute(), dfi.compute()])
        dfa = SingPartFrame(dfa)
        return dfa 
    
    def plot_yproj(self, start: int=0, end: int=-1, interval: int=1, hdf: Callable[[dd.DataFrame],dd.DataFrame]=lambda df: df):
        """
        Plot projection along y-axis for files every interval 
        """
        t0 = ParticleFrame(self.ensemble[start])['time'].min().compute()
        t1 = ParticleFrame(self.ensemble[end-1])['time'].max().compute()
        fig = go.Figure()
        for f in self.ensemble[start:end:interval]:
            df = hdf(ParticleFrame(f))
            if len(df.index)==0:
                print(f"Reach the ParticleFrame with no data: {f}")
                break
            # Create a text column that combines the information
            df['hover_text'] = df.apply(lambda row: f"<b>Time:</b> {row['time']}<br>"
                                                f"<b>Index:</b> {row['index']}<br>"
                                                f"<b>Energy:</b> {row['usrpl01']}<br>"
                                                f"<b>X:</b> {row['x1']}<br>"
                                                f"<b>Y:</b> {row['x2']}<br>"
                                                f"<b>Z:</b> {row['x3']}<br>",
                                                axis=1, meta=('hover_text', 'object'))                          
            df_subset = df[columns_to_extrat+['hover_text']].compute()
            time = df_subset['time'].min()
            fig.add_trace(go.Scatter(
                x=df_subset['x1'],
                y=df_subset['x3'],
                mode='markers',
                marker=dict(
                    color=df_subset['time'],
                    colorscale='Jet',
                    colorbar=dict(title='Time'),
                    cmin=t0, cmax=t1
                ),
                text=df_subset['hover_text'],
                hoverinfo='text',
                hovertemplate='%{text}<extra></extra>',
                name=f'Time:{time}'
            ))
        fig.update_layout(
            coloraxis_colorbar=dict(
                x=0.95, y=0.5, len=0.75, thickness=20
            ),
            legend=dict(
                x=1.35, y=1, xanchor='right', yanchor='top'
            ),
            title=f'Scatter Plot of Particles from {start} to {end} with interval: {interval}',
            xaxis_title='X Axes',
            yaxis_title='Z Axis',
        )
        buttons = [
            dict(label='Hide', method='restyle',
                args=[{'visible': ["legendonly"]*(len(fig.data))}]),  # Sets all traces to not visible
            dict(label='Show', method='restyle',
                args=[{'visible': ["True"]*len(fig.data)}])   # Sets all traces to visible
        ]
        # Update layout with buttons
        fig.update_layout(
            width=1000,
            height=800,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=buttons,
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=1.1, xanchor="left",
                    y=1.2, yanchor="top"
                )
            ])
        fig.show()

def doc_main():
    doctest.testmod(verbose=False)

if __name__ == "__main__":
    doc_main()