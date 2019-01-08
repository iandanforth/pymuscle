import numpy as np
from pymuscle import PotvinFuglevandMuscle as Muscle
import plotly.graph_objs as go
from plotly.offline import plot
from util import timing


@timing
def instantiation(a, n):
    for _ in range(a):
        m = Muscle(n)


@timing
def step(a, n):
    moderate_input = 40.0
    in_vec = np.full(n, moderate_input)
    m = Muscle(n)
    for _ in range(a):
        m.step(in_vec, 1.0)


def main():
    step_durations = []
    instantiation_durations = []
    loops = 10000
    exps = list(range(10, 16))
    for i in exps:
        n = 2 ** i
        inst_dur, _ = instantiation(loops, n)
        instantiation_durations.append(inst_dur)
        step_dur, _ = step(loops, n)
        step_durations.append(step_dur)

    instantiation_durations = np.log(instantiation_durations)
    step_durations = np.log(step_durations)

    # Log plot of durations by exponents

    layout = go.Layout(
        title="Log Plot of Durations for N=2^X",
        xaxis={"title": "Number of Motor Units - N=2^X"},
        yaxis={"title": "Duration (log(seconds)) of 10000 loops"}
    )
    data = [
        {'y': instantiation_durations, 'x': exps, 'title': 'Instantiation'},
        {'y': step_durations, 'x': exps, 'title': '.step()'}
    ]
    fig = go.Figure(
        data=data,
        layout=layout
    )
    plot(fig)


if __name__ == '__main__':
    main()
