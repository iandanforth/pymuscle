import numpy as np
import plotly.graph_objs as go
import colorlover as cl
from numpy import ndarray
from plotly.offline import plot


class PotvinChart(object):

    def __init__(self, time_by_forces: ndarray, step_size: float):

        forces_by_time = np.array(time_by_forces).T
        motor_unit_count, steps = forces_by_time.shape
        sim_duration = steps * step_size
        times = np.arange(0.0, sim_duration, step_size)

        # Setting colors for plot.
        potvin_scheme = [
            'rgb(115, 0, 0)',
            'rgb(252, 33, 23)',
            'rgb(230, 185, 43)',
            'rgb(107, 211, 100)',
            'rgb(52, 211, 240)',
            'rgb(36, 81, 252)',
            'rgb(0, 6, 130)'
        ]

        # It's hacky but also sorta cool.
        c = cl.to_rgb(cl.interp(potvin_scheme, motor_unit_count))
        c = [val.replace('rgb', 'rgba') for val in c]
        c = [val.replace(')', ',{})') for val in c]

        # Assing non-public attributes
        self._step_size = step_size
        self._forces_by_time = forces_by_time
        self._c = c
        self._times = times

        # Assign public attributes
        self.motor_unit_count = motor_unit_count

    def _get_color(self, trace_index: int) -> str:
        # The first and every 20th trace should be full opacity
        alpha = 0.2
        if trace_index == 0 or ((trace_index + 1) % 20 == 0):
            alpha = 1.0
        color = self._c[trace_index].format(alpha)
        return color

    def display(self) -> None:
        # Per Motor Unit Force
        data = []
        for i, t in enumerate(self._forces_by_time):
            trace = dict(
                x=self._times,
                y=t,
                name=i + 1,
                marker=dict(
                    color=self._get_color(i)
                ),
            )
            data.append(trace)

        layout = dict(
            title='Motor Unit Forces by Time',
            yaxis=dict(
                title='Motor unit force (relative to MU1 tetanus)'
            ),
            xaxis=dict(
                title='Time (s)'
            )
        )

        fig = dict(
            data=data,
            layout=layout
        )
        plot(fig, filename='forces-by-time.html', validate=False)
