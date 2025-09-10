import numpy as np

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
    """A rotating rigid body"""
    def __init__(
            self,
            I_body: np.ndarray,                                      # Rotational inertia along each basis vector in the body frame (assuming diagonal inertia tensor)
            R0_inertial: np.ndarray = np.eye(3),                     # Initial rotation matrix, in inertial frame
            omega0_body: np.ndarray = np.array([0,0,np.radians(5)])  # Initial angular velocity vector, in body frame, in units of rad/s
        ):
        self.I = I_body
        assert np.isclose(np.linalg.det(R0_inertial), 1)
        self.R = R0_inertial
        self.omega = omega0_body

    def set_input(self, R_inertial: np.ndarray, omega_body: np.ndarray) -> None:
        self.R = R_inertial
        self.omega = omega_body
    
    def get_derivs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the derivative of the body rotation matrix and the body angular velocity vector"""
        # Angular momentum
        # Note: though this is free rotation i.e. angular momentum is consevred,
        # the angular momentum vector in the body frame is not constant
        # (though it is constant in any inertial frame).
        L_body = self.I @ self.omega
        # Euler equations: dω/dt in the body frame (which is why L is not a constant)
        omegadot_body = np.linalg.inv(self.I) @ np.cross(L_body, self.omega)
        # Orientation derivative: dR/dt = R * skew(ω in inertial coords) = R * skew(R @ ω)
        Rdot_inertial = self.R @ skew(self.R @ self.omega)
        return Rdot_inertial, omegadot_body
