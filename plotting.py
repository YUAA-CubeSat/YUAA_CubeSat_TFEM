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
    directions = [
        (1, 0, 0),  # x-axis
        (0, 1, 0),  # y-axis
        (0, 0, 1)   # z-axis
    ]
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

    # --- Inertial Frame Axes ---
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(1.2, 1.2, 1.2)
    axesWidget = vtk.vtkOrientationMarkerWidget()
    axesWidget.SetOrientationMarker(axes)
    axesWidget.SetInteractor(interactor)
    axesWidget.EnabledOn()
    axesWidget.InteractiveOff()
    axesWidget.SetViewport(0.0, 0.0, 0.3, 0.3)
    
    # --- Create Angular Velocity Vector Visualization ---
    # Create a line to represent angular velocity
    omegaSource = vtk.vtkArrowSource()
    omegaSource.SetTipResolution(20)
    omegaSource.SetShaftResolution(20)
    
    # Create transform for the omega vector
    omegaTransform = vtk.vtkTransform()
    
    # Create transform filter for the arrow
    omegaTransformFilter = vtk.vtkTransformPolyDataFilter()
    omegaTransformFilter.SetInputConnection(omegaSource.GetOutputPort())
    omegaTransformFilter.SetTransform(omegaTransform)
    omegaTransformFilter.Update()
    
    # Create mapper and actor for omega vector
    omegaMapper = vtk.vtkPolyDataMapper()
    omegaMapper.SetInputConnection(omegaTransformFilter.GetOutputPort())
    
    omegaActor = vtk.vtkActor()
    omegaActor.SetMapper(omegaMapper)
    omegaActor.GetProperty().SetColor(1.0, 1.0, 0.0)  # Yellow color
    renderer.AddActor(omegaActor)
    
    # Text to display angular velocity magnitude
    omegaTextActor = vtk.vtkTextActor()
    omegaTextActor.SetInput("ω = 0.0 rad/s")
    omegaTextActor.GetTextProperty().SetFontSize(16)
    omegaTextActor.GetTextProperty().SetColor(1.0, 1.0, 0.0)  # Yellow color
    omegaTextActor.SetPosition(10, 10)
    renderer.AddActor2D(omegaTextActor)
    
    renderer.ResetCamera()

    # Create a transform for the body
    transform = vtk.vtkTransform()
    
    # Add a simple observer to handle key presses
    def on_key_press(obj, event):
        key = obj.GetKeySym()
        print(f"Key pressed: {key}")
        if key == 'q':
            obj.ExitCallback()

    interactor.AddObserver('KeyPressEvent', on_key_press)
    
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
        
        # Create and update the transformation matrix
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
        
        # Update the transform
        transform.SetMatrix(matrix)
        
        # Apply to all actors
        cubeActor.SetUserTransform(transform)
        for actor in body_axes_actors:
            actor.SetUserTransform(transform)
            
        # Update angular velocity visualization
        # Scale the vector for better visibility
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 0:
            omega_dir = omega / omega_norm
            scale_factor = min(1.0, max(0.2, omega_norm * 10))  # Scale based on magnitude but limit range
            
            # Create a new transform for omega that includes rotation and scaling
            omegaTransform.Identity()  # Reset transform
            
            # First rotate to align with omega direction
            # Calculate rotation from z-axis (default arrow direction) to omega direction
            if abs(omega_dir[2] - 1.0) < 1e-6:  # omega is already aligned with z-axis
                pass  # No rotation needed
            elif abs(omega_dir[2] + 1.0) < 1e-6:  # omega is opposite to z-axis
                omegaTransform.RotateX(180)  # Rotate 180 degrees around X
            else:
                # Cross product to get rotation axis
                axis = np.cross([0, 0, 1], omega_dir)
                axis = axis / np.linalg.norm(axis)
                
                # Dot product to get rotation angle
                angle = np.arccos(omega_dir[2])  # dot product of [0,0,1] and omega_dir
                
                # Apply rotation
                omegaTransform.RotateWXYZ(np.degrees(angle), axis[0], axis[1], axis[2])
            
            # Then scale according to magnitude
            omegaTransform.Scale(scale_factor, scale_factor, scale_factor)
            
            # Update the transform filter
            omegaTransformFilter.Update()
            
            # Update the text display
            omegaTextActor.SetInput(f"ω = {omega_norm:.3f} rad/s")
        
        # Apply the body's transform to the omega actor to place it at the same origin
        omegaActor.SetUserTransform(transform)
        
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
