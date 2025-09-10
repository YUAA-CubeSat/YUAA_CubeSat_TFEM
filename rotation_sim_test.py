from rotation_sim import *
from constants import *
from plotting import animate_body_rotation_vtk
import scipy

def rotation_deriv(t, state_vec, body: Body):
    """Derivative function for Body rotation only.
    state_vec: shape (12,) = [R (9), omega (3)]
    Returns concatenated [Rdot.flatten(), omegadot]
    """
    R = state_vec[:9].reshape(3,3)
    omega = state_vec[9:12]
    body.set_input(R, omega)
    Rdot, omegadot = body.get_derivs()
    return np.concatenate([Rdot.flatten(), omegadot])

def propagate_rotation(body: Body, t_span, t_eval=None, **kwargs):
    """Propagate the rotation of a Body using solve_ivp.
    t_span: (t0, tf)
    t_eval: array of times to evaluate solution at (optional)
    Returns the result from solve_ivp.
    """
    y0 = np.concatenate([body.R.flatten(), body.omega])
    sol = scipy.integrate.solve_ivp(
        lambda t, y: rotation_deriv(t, y, body),
        t_span,
        y0,
        t_eval=t_eval,
        **kwargs
    )
    return sol

omega0 = np.array([1,1,1]) * np.radians(5)
R0 = np.eye(3)
body = Body(
    np.diag([CUBESAT_ROTI_XY.to(u.kg*u.m**2).magnitude, CUBESAT_ROTI_XY.to(u.kg*u.m**2).magnitude, CUBESAT_ROTI_Z.to(u.kg*u.m**2).magnitude]),
    R0_inertial = R0,
    omega0_body=omega0
)

prop_duration_sec = 600
sol = propagate_rotation(body, (0,prop_duration_sec), dense_output=True)
animate_body_rotation_vtk(sol, fps=20)