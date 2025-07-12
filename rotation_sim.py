import numpy as np

from lumped_mass import FlatLumpedMass

def reorthonormalize(R: np.ndarray) -> np.ndarray:
    """Clip a given matrix to the nearest orthonormal matrix using SVD"""
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt

def skew(w):
    """Return the 3x3 skew matrix of a 3-vector"""
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

class Body:
    """Dataclass describing a rotating rigid body composed of lumped masses"""
    def __init__(
            self,
            lumped_masses: list[FlatLumpedMass],
            I: np.ndarray,                                      # Rotational inertia along each basis vector (assuming diagonal inertia tensor)
            R0: np.ndarray = np.eye(3),                         # Initial rotation matrix
            omega0: np.ndarray = np.array([0,0,np.radians(5)])  # Initial angular velocity vector
        ):
        self.elts = lumped_masses
        self.I = I
        assert np.isclose(np.linalg.det(R0), 1)
        self.R = R0
        self.omega = omega0

    def get_derivs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the derivative of the body rotation matrix and the body angular velocity vector"""
        # Angular momentum
        # Note: though this is free rotation i.e. angular momentum is consevred,
        # the angular momentum vector in the body frame is not constant
        # (though it is constant in any inertial frame).
        body_L = self.I @ self.omega
        # Euler equations: dω/dt in the body frame (which is why L is not a constant)
        omegadot = np.linalg.inv(self.I) @ np.cross(body_L, self.omega)
        # Orientation derivative: dR/dt = R * skew(ω)
        body_Rdot = self.R @ skew(self.omega)
        return body_Rdot, omegadot
