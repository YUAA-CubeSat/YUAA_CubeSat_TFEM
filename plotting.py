import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lumped_mass import LumpedMass

def plot_elt_T_q_flows(ts: np.ndarray, Ts: np.ndarray, elt: LumpedMass, ax=None, orbital_period=None):
    if ax is None:
        fig, ax = plt.subplots()
    q_flows = elt.get_history(ts).filter(like='q_')

    if orbital_period:
        ts /= orbital_period

    artists = []

    axT = ax
    T_line, = axT.plot(ts,Ts, label='T')
    artists.append(T_line)
    if orbital_period:
        axT.set_xlabel('Time (orbital periods)')
    elif type(ts[0]) == float:
        axT.set_xlabel('Time (s)')
    else:
        axT.set_xlabel('Time')
    axT.set_ylabel('T (K)')

    axQ = axT.twinx()
    for lbl, qs in q_flows.items():
        q_scatter = axQ.scatter(ts, qs, label=lbl)
        artists.append(q_scatter)
    axQ.set_ylabel('Heat flux (W)')

    # Add legend last so it's on top of everything,
    # and explicitly pass it all artists from both axes
    axQ.legend(handles=artists)

    return ax