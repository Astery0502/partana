from ParticleFrames import ParticleFrames
from typing import Union, List, Tuple, Callable, Iterable

import h5py
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ParticlePlots():
    def __init__(self,cols):
        self.cols = cols

    @staticmethod
    def rebin_val(val: Iterable, bin:int) -> np.ndarray:
        val = np.array(val)
        padding_size = (bin - len(val) % bin) % bin
        padded_data = np.pad(val, (0, padding_size), 'constant')
        rebinned_val = padded_data.reshape(-1, bin).sum(axis=1)
        return rebinned_val

    def plot_bottom(self, df, bins:int=10, cols:Union[Iterable[str],None]=None):
        """
        Here we note that the function can only be use for bottom escape particles
        """
        if cols is None:
            cols = self.cols
        ti = cols.index('time'); ii = cols.index('index'); eki = cols.index('usrpl01')
        x1i = cols.index('x1'); x2i = cols.index('x2')
        hover_text = np.array([f"<b>Time:</b> {row[ti]}<br>"
                        f"<b>Index:</b> {row[ii]}<br>"
                        f"<b>Energy:</b> {row[eki]}<br>"
                        f"<b>X:</b> {row[x1i]}<br>"
                        f"<b>Y:</b> {row[x2i]}<br>" for row in df]).reshape(-1,1)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=False, 
                vertical_spacing=0.1, subplot_titles=('Bottom Ejection', 'Energy Histogram'))
        fig.add_trace(go.Scatter(
            x=df[:,x1i],
            y=df[:,x2i],
            mode='markers',
            marker=dict(
                color=df[:,eki]*511,
                colorscale='Jet',
                colorbar=dict(title='Ek (keV)'),
            ),
            text=hover_text,
            hoverinfo='text',
            hovertemplate='%{text}<extra></extra>',
        ), row=1, col=1)
        fig.update_xaxes(range=[-0.8,0.8], row=1, col=1)
        ek_hists = ParticleFrames.get_general_hists(ParticleFrames(""),eki)(df, bins=bins)
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
        fig.update_xaxes(title_text='Ek (keV)', row=2, col=1)
        fig.update_yaxes(title_text='Density', row=2, col=1)
        fig.show()
    
    def plot_traj(self, df, cols:Union[Iterable[str],None]=None, bin: int=1):
        """
        Here we note that it is only applied to single particle trajectory df
        """
        # Add a hover text column to the DataFrame for the plot: time; index; energy; x1; x2; x3
        if cols is None:
            cols = self.cols
        ti = cols.index('time'); ii = cols.index('index'); eki = cols.index('usrpl01')
        x1i = cols.index('x1'); x2i = cols.index('x2'); x3i = cols.index('x3')
        u1i = cols.index('u1'); dti = cols.index('dt'); gfi = cols.index('usrpl02')
        pl4i = cols.index('pl04'); pl5i = cols.index('pl05'); pl6i = cols.index('pl06')
        hover_text = np.array([f"<b>Time:</b> {row[ti]}<br>"
                        f"<b>Index:</b> {row[ii]}<br>"
                        f"<b>Energy:</b> {row[eki]}<br>"
                        f"<b>gfunc:</b> {row[gfi]}<br>"
                        f"<b>X:</b> {row[x1i]}<br>"
                        f"<b>Y:</b> {row[x2i]}<br>" 
                        f"<b>Z:</b> {row[x3i]}<br>" for row in df]).reshape(-1,1)
        #index = self['index'][0].compute()
        time = df[:,ti][::bin]

        fig = make_subplots(
            rows = 1, cols = 2,
            specs=[[{'type': 'scatter3d'},
                   {'type': 'scatter'}]],
            subplot_titles=(f'Trajectory of Particle'),

        )

        fig.add_trace(go.Scatter3d(
            x = df[:,x1i][::bin],
            y = df[:,x2i][::bin],
            z = df[:,x3i][::bin],
            mode='markers',
            marker=dict(
                size=5,
                color=time,
                colorscale='Jet',
                opacity=0.9
            ),
            text=hover_text[::bin],
            hoverinfo='text',
            hovertemplate='%{text}<extra></extra>',
            name='Trajectory'
        ), row=1, col=1)

        # calculate dek values
        dek1 = df[:,pl4i] * df[:,u1i] * df[:,dti]
        dek2 = df[:,pl5i] * df[:,u1i] * df[:,dti]
        dek3 = df[:,pl6i] * df[:,u1i] * df[:,dti]
        print("dek1:", dek1.sum())
        print("dek2:", dek2.sum())
        print("dek3:", dek3.sum())

        # Add the time series scatter plots
        for i, u in enumerate([dek1,dek2,dek3], start=1):
            dek = self.rebin_val(u, bin)
            fig.add_trace(go.Scatter(x=time, y=dek, 
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=time,
                            colorscale='Jet',
                            opacity=0.9
                            ),
                        text=hover_text[::bin],
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

    def plot_yproj(self, pfs: ParticleFrames):
        """
        Plot projection along y-axis for files every interval 
        """
        ds_names = pfs.ds_names
        fig = go.Figure()
        with h5py.File(pfs.h5, 'r') as hdf:
            ti = pfs.col2num('time'); ii = pfs.col2num('index'); eki = pfs.col2num('usrpl01')
            x1i = pfs.col2num('x1'); x2i = pfs.col2num('x2'); x3i = pfs.col2num('x3'); dti = pfs.col2num('dt')

            t0 = hdf[ds_names[0]][:,ti].min(); t1 = hdf[ds_names[-1]][:,ti].max()
            print('preparation')
            for ds in ds_names:
                dsi = hdf[ds]
                if dsi.shape[0]==0:
                    print(f"Reach the ParticleFrame with no data: {ds}")
                    break
                print("end data extraction")
                # Create a text column that combines the information
                hover_text = np.array([f"<b>Time:</b> {row[ti]}<br>"
                                f"<b>Index:</b> {row[ii]}<br>"
                                f"<b>Energy:</b> {row[eki]}<br>"
                                f"<b>X:</b> {row[x1i]}<br>"
                                f"<b>Y:</b> {row[x2i]}<br>" 
                                f"<b>Z:</b> {row[x3i]}<br>" for row in dsi]).reshape(-1,1)
                print("end hover text creation")
                time = dsi[:,ti].min()
                fig.add_trace(go.Scatter(
                    x=dsi[:,x1i],
                    y=dsi[:,x3i],
                    mode='markers',
                    marker=dict(
                        color=dsi[:,ti],
                        colorscale='Jet',
                        colorbar=dict(title='Time'),
                        cmin=t0, cmax=t1
                    ),
                    text=hover_text,
                    hoverinfo='text',
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Time:{time}'
                ))
                print('one loop ends')
            print("end traces")
            fig.update_layout(
                coloraxis_colorbar=dict(
                    x=0.95, y=0.5, len=0.75, thickness=20
                ),
                legend=dict(
                    x=1.35, y=1, xanchor='right', yanchor='top'
                ),
                title=f'Scatter Plot of Particles from time {t0} to {t1}', 
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
            print("end fig setup")
            fig.show()