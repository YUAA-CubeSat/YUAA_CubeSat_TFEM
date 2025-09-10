from abc import ABC, abstractmethod
import enum
import numpy as np
import pandas as pd
import pint; from pint import Quantity; u=pint.UnitRegistry()

import constants

class Rad_Src(enum.Enum):
    SUN = enum.auto()
    EARTH = enum.auto()

class LumpedMass(ABC):
    """Generic lumped mass abstract base class.
    
    Any lumped mass has a mass, specific heat capacity, 
    radiative area and accompanying emissivity and absorptance, 
    an internal heat source, and a temperature.

    A lumped mass also has methods to calculate the change in temperature given a heat flow,
    and inflowing and outflowing radiative heat
    given a concrete implementation of get_view_factor.

    The store_history parameter to __init__ specifies whether this LumpedMass
    should track the results of calls to its methods. 
    When used for numerical integration, 
    LumpedMass.set_input and LumpedMass.find_Tdot should each be called once at every evaluation point.
    If store_history is True, set_input will initialize an empty snapshot,
    methods of LumpedMass can record calculation results into the snapshot as they please,
    and finally find_Tdot will save the snapshot to an internal list
    along with the corresponding time.
    Use get_history to retrieve the recorded history for a given set of times
    (which should be the set of times returned by the numerical integrator).
    """

    def __init__(
            self, 
            m: Quantity,                    # mass
            c: Quantity,                    # specific heat capacity
            A_rad: Quantity,                # radiative area
            a: float,                       # absorptivity
            e: float,                       # emissivity
            q_int: Quantity,                # internal heating power
            T0: Quantity,                   # initial temperature
            store_history : bool = False    # whether to store a snapshot of calculated results at every evaluation
    ):
        self.m = m.to(u.kg).magnitude
        self.c = c.to(u.J/(u.kg * u.degK)).magnitude
        self.A_rad = A_rad.to(u.m**2).magnitude
        self.a = a
        self.e = e
        self.q_int = q_int.to(u.W).magnitude
        self.T = T0.to(u.degK).magnitude

        self.sb_sigma = constants.SB_SIGMA.to(u.W/(u.m**2 * u.degK**4)).magnitude

        self.store_history = store_history
        if store_history:
            self.ts = []
            self.snapshots = []

    def set_input(self, t: float, T: float, *args, q_solar_normal: float, q_earth_normal: float, **kwargs) -> None:
        self.t = t
        self.T = T
        self.q_solar_normal = q_solar_normal
        self.q_earth_normal = q_earth_normal
        if self.store_history:
            self.snapshot = dict()

    @abstractmethod
    def _get_view_factor(self, src: Rad_Src) -> float:
        """Calculate the view factor between this element and a given radiation source"""
        ...

    def _get_q_in(self, src: Rad_Src, normal_power: float) -> float:
        return normal_power * self._get_view_factor(src) * self.a * self.A_rad

    def _get_q_out(self) -> float:
        # Heat flowing out is negative
        return -self.sb_sigma * self.e * self.T**4 * self.A_rad
    
    def _get_q_flows(self) -> np.ndarray:
        """Calculate all heat flows in the current state
        
        This generic implementation expects the normal solar and Earth heat fluxes
        to have already been stored on self from set_input,
        and returns: 
        solar direct heat in, Earth albedo + IR heat in, internal heat generation, heat radiated out (negative)
        """
        q_solar = self._get_q_in(Rad_Src.SUN, self.q_solar_normal)
        q_alb_IR = self._get_q_in(Rad_Src.EARTH, self.q_earth_normal)
        q_out = self._get_q_out()
        if self.store_history:
            self.snapshot['q_solar'] = q_solar
            self.snapshot['q_alb_IR'] = q_alb_IR
            self.snapshot['q_int'] = self.q_int
            self.snapshot['q_out'] = q_out
        return np.array([q_solar, q_alb_IR, self.q_int, q_out])
    
    def _get_Tdot(self, q_net: float) -> float:
        return q_net / (self.m * self.c)
    
    def find_Tdot(self) -> float:
        q_flows = self._get_q_flows()
        Tdot = self._get_Tdot(q_flows.sum())
        if self.store_history:
            self.ts.append(self.t)
            self.snapshots.append(self.snapshot)
        return Tdot
    
    def get_history(self, ts: np.ndarray) -> pd.DataFrame:
        """Return recorded calculation results for a given set of times
        
        Filters the recorded snapshots for those whose times are closest
        to the times given in ts,
        and returns a DataFrame with a time index.
        The filtration is necessary because the numerical integrator 
        will probably evaluate at more time points than it ends up returning.
        """
        history = pd.DataFrame(
            index=pd.Index(self.ts, name='t'),
            data=self.snapshots,
        )
        history.sort_index(inplace=True)
        insertion_indices = np.searchsorted(history.index.values, ts)
        # searchsorted returns, for each time in ts, 
        # the index at which it would have to be inserted into history to maintain ascending order.
        # Either the history entry to the left or to the right may be closer in time.
        for i in range(len(insertion_indices)):
            match_idx = insertion_indices[i]
            left = history.index.values[match_idx-1 if match_idx > 0 else match_idx]
            right = history.index.values[match_idx]
            if abs(ts[i] - left) < abs(ts[i] - right):
                insertion_indices[i] = match_idx - 1
        return history.iloc[insertion_indices]


class UniformLumpedMass(LumpedMass):
    """A lumped mass that always has a constant view factor to everything"""

    def __init__(
            self, 
            *args, 
            F: float, # Constant view factor
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.F = F 

    def _get_view_factor(self, src: Rad_Src) -> float:
        return self.F
    

class FlatLumpedMass(LumpedMass):
    """Lumped mass with a one-sided flat area that properly calculates view factors"""

    def __init__(
            self, 
            *args,
            body_normal: np.ndarray = np.array([1,0,0]), # Normal to this lumped mass's surface in the body reference frame
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.body_normal = body_normal / np.linalg.norm(body_normal)

    def set_input(self, *args, r_earth: float, r_sun: float, body_R: np.ndarray, **kwargs) -> None:
        """
        Parameters:
            r_earth - 3-vector from this mass to the Earth in the inertial reference frame
            r_sun - 3-vector from this mass to the Sun in the inertial reference frame
            body_R - 3x3 rotation matrix of the body in the inertial reference frame
            Vector magnitudes don't matter.
        """
        super().set_input(*args, **kwargs)
        self.r_earth = r_earth
        self.r_sun = r_sun
        self.body_R = body_R

    @property
    def normal(self):
        """Normal to this lumped mass's surface in the inertial reference frame"""
        return self.body_R @ self.body_normal.T

    def _get_view_factor(self, src: Rad_Src) -> float:
        if src == Rad_Src.EARTH:
            vec_to_rad_src = self.r_earth
        elif src == Rad_Src.SUN:
            vec_to_rad_src = self.r_sun
        vec_to_rad_src_hat = vec_to_rad_src/np.linalg.norm(vec_to_rad_src)
        dot = np.dot(vec_to_rad_src_hat, self.normal)
        if dot < 0: return 0 # Assume one-sided surface
        else: return dot

class ConnectedLumpedMass(FlatLumpedMass):
    """A lumped mass with contact conductances to adjacent lumped masses"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connections = []

    def connect(
            self, 
            other: 'ConnectedLumpedMass', 
            K: Quantity                     # contact heat conductance
    ) -> None:
        """Bidirectional heat conduction connection
        
        Call only once for a pair of lumped masses!
        """
        self.connections.append((other, K.to(u.W/u.degK).magnitude))
        other.connections.append((self, K.to(u.W/u.degK).magnitude))

    def _get_q_cond_net(self) -> float:
        """Calculates the net heat transfer from all the contact conductances"""
        q_cond_net = 0
        for elt, K in self.connections:
            q_cond_net += (elt.T - self.T)*K
        return q_cond_net

    def _get_q_flows(self) -> np.ndarray:
        q_cond_net = self._get_q_cond_net()
        if self.store_history:
            self.snapshot['q_cond_net'] = q_cond_net
        return np.concat((super()._get_q_flows(), np.array([q_cond_net])))
    