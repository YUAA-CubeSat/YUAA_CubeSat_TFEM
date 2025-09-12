import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lumped_mass import LumpedMass

def plot_elt_T_q_flows(ts: np.ndarray, Ts: np.ndarray, elt: LumpedMass, ax=None, orbital_period=None, show_legend=True):
    """Plot heat flows for a LumpedMass
    
    elt must have history enabled, heat flows will be taken from there.
    ts: 1D array of times
    Ts: 1D array of temperatures
    orbital_period: optional, x-axis will be labeled with ts/orbital_period if supplied
    """
    if ax is None:
        fig, ax = plt.subplots()
    q_flows = elt.get_history(ts).filter(like='q_')

    if orbital_period:
        ts = ts.copy() / orbital_period

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
    if show_legend:
        axQ.legend(handles=artists)

    return axT, axQ, artists

def plot_elts_T_q_flows(ts: np.ndarray, Ts: np.ndarray, elts: list[LumpedMass], orbital_period=None, ncols=2, elt_names: list[str]|None = None):
    """
    Plot T and q_flows for multiple LumpedMass objects in a grid of subplots.
    ts: 1D array of times
    Ts: 2D array of temperatures, shape (n_elts, len(ts))
    elts: list of LumpedMass objects
    orbital_period: optional, x-axis will be labeled with ts/orbital_period if supplied
    ncols: number of columns in the subplot grid
    elt_names: optional list of element names to act as subplot titles
    """
    import math
    n_elts = len(elts)
    nrows = math.ceil(n_elts / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)

    # Determine global x/y limits for all subplots
    all_Ts = np.array(Ts).flatten()
    T_min, T_max = np.min(all_Ts), np.max(all_Ts)
    if orbital_period:
        ts_scaled = ts / orbital_period
        x_min, x_max = np.min(ts_scaled), np.max(ts_scaled)
    else:
        x_min, x_max = np.min(ts), np.max(ts)
    
    # Store all q_flow data to determine y-axis limits for heat flux
    all_q_flows = []
    
    # First pass to collect all heat flux data
    for elt in elts:
        q_flows = elt.get_history(ts).filter(like='q_')
        for _, qs in q_flows.items():
            all_q_flows.extend(qs.tolist())
    
    # Calculate global heat flux limits
    q_min, q_max = np.min(all_q_flows), np.max(all_q_flows)
    # Add a small margin
    q_range = q_max - q_min
    q_min -= 0.05 * q_range
    q_max += 0.05 * q_range

    # Store all artists for the common legend
    all_artists = []

    # Create plots using plot_elt_T_q_flows
    for idx, (elt, Ts_elt) in enumerate(zip(elts, Ts)):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        
        # Use plot_elt_T_q_flows but don't show individual legends
        axT, axQ, artists = plot_elt_T_q_flows(ts, Ts_elt, elt, ax=ax, orbital_period=orbital_period, show_legend=False)
        
        # Configure axes based on position
        # Only show x label on bottom row
        if row != nrows - 1:
            axT.set_xlabel("")
        
        # Only show primary y label on leftmost column
        if col != 0:
            axT.set_ylabel("")
        
        # Only show secondary y-axis (heat flux) on rightmost column
        if col != ncols - 1:
            axQ.set_yticks([])
            axQ.set_ylabel("")
        
        # Set consistent limits
        axT.set_ylim(T_min, T_max)
        axT.set_xlim(x_min, x_max)
        axQ.set_ylim(q_min, q_max)
        
        # Save artists from the first plot for the legend
        if idx == 0:
            all_artists = artists
            
        # Set subplot title
        ax.set_title(f"Element {idx}" if elt_names is None else elt_names[idx])

    # Hide unused axes
    for idx in range(n_elts, nrows*ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Create a single legend outside the subplots
    fig.legend(handles=all_artists, loc='lower center', bbox_to_anchor=(0.5, 0), 
               ncol=min(len(all_artists), 3), frameon=True)
        
    # Adjust layout to make room for the legend
    fig.tight_layout(rect=(0, 0.1, 1, 0.98))
    
    return fig, axes

def plot_elts_Ts(ts: np.ndarray, Ts: np.ndarray, ax = None, orbital_period = None, elt_names: list[str]|None = None):
    """Plot temperatures over time
    
    ts: 1D array of times
    Ts: 2D array of temperatures, shape (n_elts, len(ts))
    orbital_period: optional, x-axis will be labeled with ts/orbital_period if supplied
    elt_names: optional list of element names for legend
    """
    if ax is None:
        fig, ax = plt.subplots()

    if orbital_period:
        ts = ts.copy() / orbital_period

    for i in range(len(Ts)):
        ax.plot(ts, Ts[i,:], label=f"Element {i}" if elt_names is None else elt_names[i])

    if orbital_period:
        ax.set_xlabel('Time (orbital periods)')
    elif type(ts[0]) == float:
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xlabel('Time')
    ax.set_ylabel('T (K)')
    ax.legend()
    return ax

def animate_body_rotation_vtk(sol, fps=50):
    """
    Animate the rotation of a rigid body using VTK.

    WARNING: vibe coded
    
    Parameters:
    - sol: The solution object from scipy.integrate.solve_ivp.
           Expects state vectors: shape (12,) = [R (9), omega (3)] 
           It's recommended to use `dense_output=True` in `solve_ivp` for a smooth animation.
    """
    import vtk
    import time
    
    # Use dense output if available for a smooth animation
    if hasattr(sol, 'sol') and callable(sol.sol):
        t_start, t_end = sol.t[0], sol.t[-1]
        times = np.arange(t_start, t_end, 1/fps)
        y_vals = sol.sol(times)
        Rs = y_vals[:9, :].T.reshape(-1, 3, 3)
        omegas = y_vals[9:12, :].T  # Extract angular velocities
    else:
        times = sol.t
        Rs = sol.y[:9, :].T.reshape(-1, 3, 3)
        omegas = sol.y[9:12, :].T  # Extract angular velocities

    # --- VTK Visualization Setup ---
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(600, 600)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)
    interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interactorStyle)
    
    renderer.SetBackground(0.1, 0.2, 0.4)
    
    # Create a cube to represent the body
    cubeSource = vtk.vtkCubeSource()
    cubeSource.SetXLength(0.5)
    cubeSource.SetYLength(0.5)
    cubeSource.SetZLength(0.5)
    
    cubeMapper = vtk.vtkPolyDataMapper()
    cubeMapper.SetInputConnection(cubeSource.GetOutputPort())
    
    cubeActor = vtk.vtkActor()
    cubeActor.SetMapper(cubeMapper)
    cubeActor.GetProperty().SetOpacity(0.7)
    renderer.AddActor(cubeActor)

    # --- Create Actors for Body Axes ---
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Red, Green, Blue for x, y, z
    body_axes_actors = []
    
    for i in range(3):
        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(20)
        arrow.SetShaftResolution(20)
        
        # Transform arrow to point in proper direction
        transform = vtk.vtkTransform()
        if i == 0:  # x-axis - already aligned
            pass
        elif i == 1:  # y-axis
            transform.RotateZ(90)
        elif i == 2:  # z-axis
            transform.RotateY(-90)
        
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputConnection(arrow.GetOutputPort())
        transformFilter.SetTransform(transform)
        transformFilter.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transformFilter.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors[i])
        actor.SetScale(0.8)  # Scale arrow size
        
        renderer.AddActor(actor)
        body_axes_actors.append(actor)

    # --- Create Actor for Omega Vector ---
    omega_arrow = vtk.vtkArrowSource()
    omega_arrow.SetTipResolution(20)
    omega_arrow.SetShaftResolution(20)

    omega_mapper = vtk.vtkPolyDataMapper()
    omega_mapper.SetInputConnection(omega_arrow.GetOutputPort())

    omega_actor = vtk.vtkActor()
    omega_actor.SetMapper(omega_mapper)
    omega_actor.GetProperty().SetColor(1, 1, 0)  # Yellow color for omega
    omega_actor.SetScale(0.5)  # Scale arrow size to 0.5
    renderer.AddActor(omega_actor)

    # --- Inertial Frame Axes ---
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(1.2, 1.2, 1.2)
    axesWidget = vtk.vtkOrientationMarkerWidget()
    axesWidget.SetOrientationMarker(axes)
    axesWidget.SetInteractor(interactor)
    axesWidget.EnabledOn()
    axesWidget.InteractiveOff()
    axesWidget.SetViewport(0.0, 0.0, 0.3, 0.3)
        
    renderer.ResetCamera()

    # Create a transform for the body
    transform = vtk.vtkTransform()
        
    # Initialize the rendering and interaction
    print(f"Animating {len(Rs)} timesteps...")
    interactor.Initialize()
    renderWindow.Render()

    # Give VTK time to set up the window
    time.sleep(0.5)
    
    # Flag to control animation loop
    animation_running = True

    # Add a callback to handle window close event
    def on_window_close(obj, event):
        nonlocal animation_running
        animation_running = False
        print("Window closed, stopping animation")
    
    renderWindow.AddObserver("DeleteEvent", on_window_close)
    
    # Instead of timer events, manually advance frames
    print("Starting manual frame-by-frame animation...")
    frame_index = 0
    
    # Start animation loop
    while animation_running and frame_index < len(Rs):
        # Get the current rotation matrix and angular velocity
        R = Rs[frame_index]
        omega = omegas[frame_index]
        
        # Create and update the transformation matrix for the body
        matrix = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                matrix.SetElement(i, j, R[i, j])
        
        # Set identity elements
        matrix.SetElement(0, 3, 0)
        matrix.SetElement(1, 3, 0)
        matrix.SetElement(2, 3, 0)
        matrix.SetElement(3, 0, 0)
        matrix.SetElement(3, 1, 0)
        matrix.SetElement(3, 2, 0)
        matrix.SetElement(3, 3, 1)
        
        # Update the transform for the body
        transform.SetMatrix(matrix)
        
        # Apply the transform to the body and its axes
        cubeActor.SetUserTransform(transform)
        for actor in body_axes_actors:
            actor.SetUserTransform(transform)
        
        # The omega vector is in the body frame. To display it in the inertial frame,
        # it must be rotated by R.
        omega_inertial = R @ omega
        
        # Create a transform for the omega vector to align it from the x-axis to the omega_inertial direction
        omega_norm = np.linalg.norm(omega_inertial)
        omega_transform = vtk.vtkTransform()
        omega_transform.Identity() # Start with an identity transform
        if omega_norm > 1e-6:
            # The default arrow source points along the x-axis
            v_from = np.array([1.0, 0.0, 0.0])
            v_to = omega_inertial / omega_norm
            
            # Find rotation axis and angle
            cross_prod = np.cross(v_from, v_to)
            dot_prod = np.dot(v_from, v_to)
            angle_rad = np.arccos(np.clip(dot_prod, -1.0, 1.0)) # clip for stability
            angle_deg = np.rad2deg(angle_rad)
            
            # If vectors are not collinear, apply rotation
            if np.linalg.norm(cross_prod) > 1e-6:
                rotation_axis = cross_prod / np.linalg.norm(cross_prod)
                omega_transform.RotateWXYZ(angle_deg, rotation_axis[0], rotation_axis[1], rotation_axis[2])
            elif dot_prod < 0: # vectors are anti-parallel
                # Rotate 180 degrees around an arbitrary axis perpendicular to v_from
                omega_transform.RotateWXYZ(180, 0, 1, 0)

        # Scale the omega vector. The scale factor determines its length in the visualization.
        # The arrow source has a default length of 1.
        scale_factor = float(omega_norm * 0.5)
        omega_transform.Scale(scale_factor, scale_factor, scale_factor)
        omega_actor.SetUserTransform(omega_transform)

        renderWindow.Render()
        interactor.ProcessEvents()  # Process any pending events
        
        if frame_index % 50 == 0:
            print(f"Frame {frame_index}/{len(Rs)}")
        
        frame_index += 1
        time.sleep(1/fps)  # Control animation speed

    print("Animation complete")
    
    # After animation completes, keep the window interactive
    if animation_running:
        print("Switching to interactive mode")
        interactor.Start()
