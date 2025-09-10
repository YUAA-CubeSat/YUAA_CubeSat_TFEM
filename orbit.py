from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import pint; from pint import Quantity; u=pint.UnitRegistry()

import constants

def check_intersection(r_sat: np.ndarray, r_sun: np.ndarray, R_earth: float) -> bool:
    """Check whether the sat is in Earth's shadow
    
    Assumes a spherical Earth with radius R_earth,
    and checks whether the line from the sat's position (r_sat)
    to the sun's position (r_sun) intersects the Earth.
    """
    # Parametrization of the sat-sun line
    # r(t) = r_sat + t * r_sun
    # We want to solve:
    # |r_sat + t * r_sun|^2 = R_earth^2
    # This expands to:
    #  t^2 |r_sun|^2 + 2t * (r_sat ⋅ r_sun) + (|r_sat|^2 - R_earth^2) = 0
    # which is a quadratic equation in t, and we want to know if it has real roots.

    # Quadratic coefficients
    a = np.dot(r_sun, r_sun)
    b = 2 * np.dot(r_sat, r_sun)
    c = np.dot(r_sat, r_sat) - R_earth**2

    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        # No intersection
        return False

    # Solve for t
    t1 = (-b + np.sqrt(discriminant)) / (2*a)
    t2 = (-b - np.sqrt(discriminant)) / (2*a)

    # Check if either t is in the range [0, 1]
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)


class Orbit(ABC):
    """Generic orbital model abstract base class.
    
    Any orbital model must calculate 
    the area-normal radiative heat flux due to the Earth or the Sun
    given a time relative to the epoch.
    """
    @abstractmethod
    def set_input(self, t) -> None:
        """Update the orbital model for a given time. Definition of time is implementation-dependent."""
        ...

    @abstractmethod
    def get_sun_earth_normal_power(self) -> tuple[float, float]:
        """Returns the area-normal solar heat flux and area-normal Earth albedo & IR heat flux for the satellite at the current time"""
        ...
    

class BetaCircularOrbit(Orbit):
    """Simple implementation of Orbit assuming uniform circular orbit using constant beta angle
    
    This orbital model simply assumes a constant solar heat inflow (+ constant Earth albedo) when the sat is in the sun,
    and constant Earth IR radiation always. 
    The orbit is defined through an altitude above the surface of the (spherical) Earth,
    an orbital period,
    and a beta angle, which is the angle between the Earth-Sun vector and the orbital plane.
    """

    def __init__(self, altitude: Quantity, period: Quantity, beta_angle: float):
        self.beta_angle = beta_angle
        self.period = period.to(u.s).magnitude
        self.beta_crit = np.arcsin(constants.EARTH_RADIUS/(constants.EARTH_RADIUS + altitude))
        self.eclipse_fraction = \
            np.arccos(
                (
                    np.sqrt((altitude ** 2 + 2 * constants.EARTH_RADIUS * altitude).to(u.m**2).magnitude)
                    /((constants.EARTH_RADIUS + altitude).to(u.m).magnitude * np.cos(beta_angle))
                )
            ) / (np.pi*u.radian) if abs(beta_angle) < self.beta_crit else 0
        self.q_solar_typ = constants.LEO_TYP_SOLAR_NORMAL_POWER.to(u.W/(u.m**2)).magnitude
        self.q_earth_ir = constants.EARTH_IR_NORMAL_POWER.to(u.W/(u.m**2)).magnitude

    def set_input(self, t: float):
        """
        Expects t in seconds
        """
        self.t= t

    def get_sun_earth_normal_power(self):
        t_mod = self.t % self.period
        if self.period/2 * (1-self.eclipse_fraction) < t_mod < self.period/2 * (1+self.eclipse_fraction):
            q_solar = 0
        else: 
            q_solar = self.q_solar_typ
        return q_solar, q_solar * constants.EARTH_ALBEDO_FACTOR + self.q_earth_ir


class SGP4Orbit(Orbit):
    """Implementation of Orbit using the SGP4 orbital propagator and JPL solar system ephemeris
    
    TEME coordinate convention, as explained in the sgp4 project documentation:
    - x & y lie in the equatorial plane
    - x is towards the vernal equinox (fixed relative to the distant stars)
    - z is towards the celestial north pole
    - the origin is at Earth's center
    - t = 0 is the epoch of the TLEs used
    """
    
    def __init__(self, tle_line1: str, tle_line2: str, ephem_kernel_path: str, t_tol: float = 0):
        """Initialize orbit with two-line elements

        ephem_kernel_path must point to a `.bsp` ehpemeris kernel file
        that will be opened with `jplephem`.
        
        t_tol specifies a time tolerance:
        the Orbit will return a cached position instead of computing a new position
        if update is called with a time within t_tol
        of the last time update queried SGP4. 
        Can be used to reduce the number of SGP4 queries at the cost of accuracy. 
        """
        from sgp4.api import Satrec
        from jplephem.spk import SPK
        super().__init__()
        self.satrec = Satrec.twoline2rv(tle_line1, tle_line2)
        self.kernel = SPK.open(ephem_kernel_path)
        self.t_tol = t_tol
        self.earth_radius = constants.EARTH_RADIUS.to(u.km).magnitude
        self.q_solar_typ = constants.LEO_TYP_SOLAR_NORMAL_POWER.to(u.W/(u.m**2)).magnitude
        self.q_earth_ir = constants.EARTH_IR_NORMAL_POWER.to(u.W/(u.m**2)).magnitude

    def set_input(self, t: pd.Timestamp):
        try:
            if abs((t-self.t).total_seconds()) < self.t_tol:
                return
        except AttributeError:
            pass
        
        # pandas allows calculation of Julian date
        # from tz-naive Timestamps, which is ambiguous
        assert t.tz is not None
        time_JD = t.to_julian_date()
        e,r_sat,v = self.satrec.sgp4(time_JD, 0.0)
        if e != 0:
            raise RuntimeError(f'SGP4 returned code {e}')
        self.t = t; self.r_sat = np.array(r_sat)

        # Get pos of Earth-Moon barycenter relative to Sol barycenter in km
        sun_to_earth_ecliptic = self.kernel[0,3].compute(time_JD) 
        # Subtract off position of Moon relative to Earth-Moon barycenter
        sun_to_earth_ecliptic -= self.kernel[3,399].compute(time_JD)

        # Rotate into the equatorial system
        # TODO - for now I'm assuming SGP4 TEME and the JPL ephemerides use the same equinox to define their x-axes, but this should ideally be verified...
        # A double negation happens:
        # 1. sun-to-earth -> earth-to-sun
        # 2. ecliptic -> equatorial
        earth_to_sun_eq = constants.ECLIPTIC_TO_EQUATORIAL_R @ sun_to_earth_ecliptic
        self.r_sun = earth_to_sun_eq

    def get_earth_sun_vecs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the vectors from the sat to the Earth and Sun

        Returns: vector from sat to Earth, vector from sat to Sun
        Both are 3-vectors in units of km.
        """
        return -self.r_sat, self.r_sun - self.r_sat

    def get_sun_earth_normal_power(self):
        if check_intersection(self.r_sat, self.r_sun, self.earth_radius):
            q_solar = 0
        else: 
            q_solar = self.q_solar_typ
        return q_solar, q_solar * constants.EARTH_ALBEDO_FACTOR + self.q_earth_ir

# TODO - one could extend SGP4Orbit to also calculate Earth's albedo and IR and maybe even the solar normal power
# depending on the sat position over the Earth, and Earth's position relative to the sun