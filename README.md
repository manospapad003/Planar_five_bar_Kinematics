# Planar Five-Bar Kinematics

A small Python package for forward and inverse kinematics of five-bar linkage mechanisms in 2D and 3D.

## Features

- Forward 2D kinematics for planar five-bar mechanisms
- Inverse 2D kinematics solver
- Forward 3D kinematics composition using planar 5-bar geometry
- Inverse 3D kinematics result object with joint-angle outputs

## Installation

This repository includes a Python package under `five_bar_functions`.

You can install it locally with:

```bash
cd five_bar_functions
python -m pip install .
```

## Usage

Use the package from the repository directly, or install it and import normally.

```python
import numpy as np
from five_bar_functions import (
    forward_2d_kinematics,
    planar_5_bar_invers_3d,
)

l0 = 31.0
l1 = 133.43
l2 = 190.40
theta1 = np.deg2rad(205)
theta2 = np.deg2rad(65)

x1, y1, x2, y2, p1, p2 = forward_2d_kinematics(theta1, theta2, l0, l1, l2)
print(f"End-effector 1: ({x1:.3f}, {y1:.3f})")

result = planar_5_bar_invers_3d(x1, y1, 0, 0, 0, 0, l0, l1, l2, 0)
print(result)
```

## API Reference

### Kinematics Functions

#### `forward_2d_kinematics(theta1, theta2, l0, l1, l2)`
Computes forward kinematics for a planar 5-bar mechanism in 2D.

**Parameters:**
- `theta1`, `theta2` (float): Joint angles in radians
- `l0` (float): Base half distance
- `l1`, `l2` (float): Link lengths

**Returns:**
- `x_1, y_1, x_2, y_2` (float): Two possible end-effector positions
- `p1, p2` (ndarray): Homogeneous transformation matrices for actuated joints

**Example:**
```python
x_1, y_1, x_2, y_2, p1, p2 = forward_2d_kinematics(
    np.deg2rad(45), np.deg2rad(60), 31.0, 133.43, 190.40
)
```

#### `inverse_2d_kinematics(x, y, l0, l1, l2)`
Solves inverse kinematics for a planar 5-bar mechanism in 2D.

**Parameters:**
- `x`, `y` (float): Desired end-effector position
- `l0`, `l1`, `l2` (float): Link parameters

**Returns:**
- Tuple of joint angles for achievable end-effector positions

#### `forward_3d_kinematics(theta1, theta2, theta3, theta4, theta5, l0, l1, l2, l3)`
Computes forward kinematics for a 3D five-bar mechanism using planar geometry.

**Parameters:**
- `theta1` to `theta5` (float): Joint angles in radians
- `l0`, `l1`, `l2` (float): Link lengths (planar)
- `l3` (float): End-effector offset

**Returns:**
- `x_5bar, y_5bar, z_5bar` (float): 5-bar mechanism end-effector position
- `x_ee, y_ee, z_ee` (float): Final end-effector position in 3D space
- `p1, p2` (ndarray): Joint transformation matrices

#### `planar_5_bar_invers_3d(x, y, z, tx, ty, tz, l0, l1, l2, l3)`
Solves inverse kinematics for the 3D five-bar mechanism.

**Parameters:**
- `x, y, z` (float): Desired end-effector position
- `tx, ty, tz` (float): Orientation angles
- `l0, l1, l2, l3` (float): Link parameters

**Returns:**
- `Planar5BarInverse3DResult`: Object containing all joint angles and intermediate calculations

### Dynamics Functions

#### `estimate_jacobian_2d(theta1, theta2, l0, l1, l2, delta=1e-6, solution=1)`
Estimates the Jacobian matrix using numerical differentiation (finite differences).

The Jacobian relates joint velocities to end-effector velocities:
```
[dx/dt]   [J11  J12] [dθ1/dt]
[dy/dt] = [J21  J22] [dθ2/dt]
```

**Parameters:**
- `theta1`, `theta2` (float): Joint angles in radians
- `l0`, `l1`, `l2` (float): Link parameters
- `delta` (float): Step size for numerical differentiation (default 1e-6)
- `solution` (int): Which end-effector solution to use (1 or 2)

**Returns:**
- `J` (ndarray): 2×2 Jacobian matrix
- `x`, `y` (float): End-effector position

**Example:**
```python
J, x, y = estimate_jacobian_2d(
    np.deg2rad(45), np.deg2rad(60), 
    31.0, 133.43, 190.40, solution=1
)
print(f"Jacobian:\n{J}")
print(f"End-effector: ({x:.3f}, {y:.3f})")
```

#### `estimate_jacobian_analytical_2d(theta1, theta2, l0, l1, l2, solution=1)`
Computes the Jacobian analytically using analytical derivatives of forward kinematics.

More accurate than numerical differentiation.

**Parameters:**
- `theta1`, `theta2` (float): Joint angles in radians
- `l0`, `l1`, `l2` (float): Link parameters
- `solution` (int): Which end-effector solution to use (1 or 2)

**Returns:**
- `J` (ndarray): 2×2 Jacobian matrix
- `x`, `y` (float): End-effector position

**Example:**
```python
J, x, y = estimate_jacobian_analytical_2d(
    np.deg2rad(45), np.deg2rad(60), 
    31.0, 133.43, 190.40
)
det_J = compute_jacobian_determinant(J)
print(f"Jacobian determinant: {det_J:.6f}")
```

#### `compute_jacobian_determinant(J)`
Computes the determinant of the Jacobian matrix.

The determinant identifies singular configurations where the mechanism loses degrees of freedom.

**Parameters:**
- `J` (ndarray): 2×2 Jacobian matrix

**Returns:**
- `det` (float): Determinant of the Jacobian

**Note:** When `det(J) = 0`, the mechanism is in a singular configuration.

**Example:**
```python
J, _, _ = estimate_jacobian_analytical_2d(theta1, theta2, l0, l1, l2)
det = compute_jacobian_determinant(J)
if abs(det) < 1e-6:
    print("Warning: Near singular configuration!")
```

#### `compute_jacobian_condition_number(J)`
Computes the condition number of the Jacobian matrix.

The condition number measures sensitivity to input perturbations. Higher values indicate worse conditioning.

**Parameters:**
- `J` (ndarray): 2×2 Jacobian matrix

**Returns:**
- `cond` (float): Condition number

**Example:**
```python
J, _, _ = estimate_jacobian_analytical_2d(theta1, theta2, l0, l1, l2)
cond = compute_jacobian_condition_number(J)
print(f"Condition number: {cond:.3f}")
if cond > 100:
    print("Configuration has poor mechanical advantage")
```

### Transformation Functions

#### `transl(x, y, z)`
Creates a homogeneous translation matrix.

#### `rotz(theta)`, `roty(theta)`, `rotx(theta)`
Create homogeneous rotation matrices about Z, Y, and X axes respectively.

#### `intersection_points_of_2c(p1, p2, r1, r2)`
Computes intersection points of two circles (used internally for kinematics).



## Examples

- `example_kinematics.py` demonstrates 2D forward kinematics and a 3D inverse verification.
- `example_3d_kinematics.py` demonstrates 3D forward kinematics and inverse validation with plotting.
- `work_space_example.py` demonstrates workspace visualization by traversing all possible motor angles and plotting the end-effector positions.
- `example_dynamics.py` demonstrates Jacobian computation, singularity analysis, and workspace conditioning visualization.

### Workspace Visualization

The `work_space_example.py` script computes and visualizes the workspace of the five-bar mechanism. It traverses all possible motor angle combinations (θ₁ and θ₂) from 0 to 2π and plots the reachable end-effector positions for both kinematic solutions.

**Features:**
- Compute workspace points by sampling motor angles at configurable resolution
- Visualize both solution branches of the mechanism's workspace
- Display a single configuration showing the linkage geometry
- Easily adjustable parameters for link lengths and resolution

**Usage:**
```bash
python work_space_example.py
```

This generates two plots:
1. A single mechanism configuration showing the linkage geometry, motor angles, and both end-effector solutions
2. The complete workspace showing all reachable positions across the full motor angle range

## Package structure

- `five_bar_functions/` — package setup and metadata
- `five_bar_functions/five_bar_functions/kinematics.py` — kinematics implementation (forward and inverse 2D/3D)
- `five_bar_functions/five_bar_functions/dynamics.py` — dynamics and Jacobian computations
- `example_kinematics.py` — 2D example script
- `example_3d_kinematics.py` — 3D example script
- `work_space_example.py` — workspace visualization example

## License

MIT License
