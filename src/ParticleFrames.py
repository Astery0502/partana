import doctest
import re

import numpy as np
import pandas as pd 

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import Union, List, Tuple, Callable, Iterable

dframe = pd.DataFrame

def rebin_val(val, bin):
    val = np.array(val)
    padding_size = (bin - len(val) % bin) % bin
    padded_data = np.pad(val, (0, padding_size), 'constant')
    rebinned_val = padded_data.reshape(-1, bin).sum(axis=1)
    return rebinned_val

class ParticleFrame(pd.DataFrame):

    @property
    def _constructor(self):
        return ParticleFrame

    def tp_prep(self) -> None:
        """
        Preprocess the DataFrame: clean column names and (adjust angles).

        Example:
        >>> import pandas as pd
        >>> df = ParticleFrame({' x1': [1,2], ' x2': [3,4]})
        >>> df.tp_prep()
        >>> df.columns.tolist()
        ['x1', 'x2']
        """
        # Clean up column names by removing spaces
        self.columns = self.columns.str.replace(" ", "", regex=True)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str):
            args = (pd.read_csv(args[0]),)
        super().__init__(*args, **kwargs)
        self.tp_prep()
    
    @staticmethod
    def get_val_range(val:str) -> Callable[[dframe,float,float], dframe]:
        def get_val_specified(df:dframe, v1:float, v2:float):
            return df[(df[val].values>=v1) & (df[val].values<=v2)]
        return get_val_specified
    
    def get_trange(self, v1:float, v2:float) -> dframe:
        """
        Get particles whose time ranges from v1 to v2        

        Example:
        >>> import pandas as pd
        >>> df = ParticleFrame({'time':[1,2,3], 'value':[2,3,5]})
        >>> (df.get_trange(1,2))['time'].tolist()
        [1, 2]
        """
        return self.get_val_range('time')(self, v1, v2)

    def get_tuntil(self, v2:float) -> dframe:
        """
        Get particles whose time ranges until v2
        """
        return self.get_trange(0, v2)

    def get_tfrom(self, v1:float) -> dframe:
        """
        Get particles whose time ranges from v1
        """
        return self.get_trange(v1, 10000)

    def get_erange(self, e0:float, e1:float, ev: bool=True) -> dframe:
        """
        Get particles whose energy ranges from e0 to e1, eV unit by default

        Example:
        >>> df = ParticleFrame({'usrpl01':[5,7,11], 'value':[1,2,3]})
        >>> (df.get_erange(7,11,ev=False))['usrpl01'].tolist()
        [7, 11]
        """
        if ev:
            e0 /= 511
            e1 /= 511
        return self.get_val_range('usrpl01')(self,e0,e1)

    def get_indexed(self, indices: Union[int, Iterable[int]]) -> dframe:
        """
        Get particles whose indices from the indices list or single int

        Example:
        >>> df = ParticleFrame({'index':[1,2,3], 'value':[1,2,3]})
        >>> (df.get_indexed([1,5]))['index'].tolist()
        [1]
        >>> (df.get_indexed(1))['index'].tolist()
        [1]
        """
        if isinstance(indices, int):
            return self[self['index'] == indices]
        return self.loc[self['index'].isin(indices)]
    
    def write_csv(self, path:str) -> None:
        return self.to_csv(path, sep=',', index=False)
    
    def get_right(self) -> dframe:
        """
        Get particles located in the positive x-axis

        Example:
        >>> df = ParticleFrame({'x1':[-1,-2,3], 'value':[1,2,3]})
        >>> (df.get_right())['x1'].tolist()
        [3]
        """
        return self.get_val_range('x1')(self,0,10000)
    
    def get_left(self) -> dframe:
        """
        Get particles located in the negative x-axis

        Example:
        >>> df = ParticleFrame({'x1':[-1,-2,3], 'value':[1,2,3]})
        >>> (df.get_left())['x1'].tolist()
        [-1, -2]
        """
        return self.get_val_range('x1')(self,-10000,0)

class DesPartFrame(ParticleFrame):
    @property
    def _constructor(self):
        return DesPartFrame

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
class SingPartFrame(ParticleFrame):
    @property
    def _constructor(self):
        return SingPartFrame

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot_traj(self, bin: int=1):
        index = self['index'].min()
        time = self['time'][::bin]
        self['hover_text'] = self.apply(lambda row: f"<b>Time:</b> {row['time']}<br>"
                                    f"<b>Index:</b> {row['index']}<br>"
                                    f"<b>Energy:</b> {row['usrpl01']}",
                                    axis=1)
        fig = make_subplots(
            rows = 1, cols = 2,
            specs=[[{'type': 'scatter3d'},
                   {'type': 'scatter'}]],
            subplot_titles=(f'Trajectory of Particle: {index}', 'Value Over Time'),

        )

        fig.add_trace(go.Scatter3d(
            x = self['x1'][::bin],
            y = self['x2'][::bin],
            z = self['x3'][::bin],
            mode='markers',
            marker=dict(
                size=5,
                color=time,
                colorscale='Jet',
                opacity=0.9
            ),
            text=self['hover_text'][::bin],
            hoverinfo='text+x+y',
            hovertemplate='%{text}<br><b>X:</b> %{x}<br><b>Z:</b> %{y}<extra></extra>',
            name='Trajectory'
        ), row=1, col=1)

        # calculate dek values
        self['dek1'] = self['pl04'] * self['u1'] * self['dt']
        self['dek2'] = self['pl05'] * self['u1'] * self['dt']
        self['dek3'] = self['pl06'] * self['u1'] * self['dt']
        print("dek1:", sum(self['dek1']))
        print("dek2:", sum(self['dek2']))
        print("dek3:", sum(self['dek3']))

        # Add the time series scatter plots
        for i, u in enumerate(['dek1', 'dek2', 'dek3'], start=1):
            dek = rebin_val(self[u], bin)
            fig.add_trace(go.Scatter(x=time, y=dek, 
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=time,
                            colorscale='Jet',
                            opacity=0.9
                            ),
                        text=self['hover_text'][::bin],
                        hoverinfo='text+x+y',
                        hovertemplate='%{text}<br><b>X:</b> %{x}<br><b>Z:</b> %{y}<extra></extra>',
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
    
    def derive_single_traj(self, index: int) -> dframe:
        """ 
        The self.ensemble is supposed to be sorted already.
        """
        df = ParticleFrame(self.ensemble[0])
        dfa = SingPartFrame(columns=df.columns).astype(df.dtypes)
        for f in self.ensemble:
            df = ParticleFrame(f)
            df_i = df.get_indexed(index)
            if df_i.empty:
                break
            dfa = SingPartFrame(pd.concat([dfa,df_i]))
        return dfa 
    
    def plot_yproj(self, start: int=0, end: int=-1, interval: int=1, hdf: Callable[[dframe],dframe]=lambda df: df):
        """
        Plot projection along y-axis for files every interval 
        """
        t0 = ParticleFrame(self.ensemble[start])['time'].min()
        t1 = ParticleFrame(self.ensemble[end-1])['time'].max()
        fig = go.Figure()
        for f in self.ensemble[start:end:interval]:
            df = hdf(ParticleFrame(f))
            # Create a text column that combines the information
            df['hover_text'] = df.apply(lambda row: f"<b>Time:</b> {row['time']}<br>"
                                        f"<b>Index:</b> {row['index']}<br>"
                                        f"<b>Energy:</b> {row['usrpl01']}",
                                        axis=1)
            time = df['time'].min()
            fig.add_trace(go.Scatter(
                x=df['x1'],
                y=df['x3'],
                mode='markers',
                marker=dict(
                    color=df['time'],
                    colorscale='Jet',
                    colorbar=dict(title='Time'),
                    cmin=t0, cmax=t1
                ),
                text=df['hover_text'],
                hoverinfo='text+x+y',
                hovertemplate='<b>Detail:</b> %{text}<br><b>X:</b> %{x}<br><b>Z:</b> %{y}<extra></extra>',
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