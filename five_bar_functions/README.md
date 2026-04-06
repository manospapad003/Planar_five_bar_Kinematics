# Five-Bar Functions

A Python package for computing 2D kinematics of five-bar linkage mechanisms.

## Overview

This package provides functions for transformation matrices and forward kinematics calculations commonly used in robotics and mechanism analysis. It includes rotation and translation transformations as well as forward kinematics solving for 2D five-bar mechanisms.

## Installation

```bash
pip install .
```

## Functions

### Transformation Matrices

#### `transl(x, y, z)`
Create a 4x4 homogeneous transformation matrix for translation.

**Parameters:**
- `x` (float): Translation in x-axis
- `y` (float): Translation in y-axis
- `z` (float): Translation in z-axis

**Returns:**
- `T` (4x4 ndarray): Homogeneous transformation matrix

#### `rotz(theta)`, `roty(theta)`, `rotx(theta)`
Create 4x4 homogeneous rotation matrices about the z, y, and x axes respectively.

**Parameters:**
- `theta` (float): Rotation angle in radians

**Returns:**
- `R` (4x4 ndarray): Homogeneous rotation matrix

### Kinematics

#### `forward_2d_kinematics(theta1, theta2, l0, l1, l2)`
Compute the end-effector position for a 2D five-bar linkage mechanism at both possible configurations.

**Parameters:**
- `theta1` (float): Angle of first actuated link (radians)
- `theta2` (float): Angle of second actuated link (radians)
- `l0` (float): Distance from origin to base pivot points
- `l1` (float): Length of actuated links
- `l2` (float): Length of connecting link

**Returns:**
- `x_1, y_1` (float, float): End-effector position for first configuration
- `x_2, y_2` (float, float): End-effector position for second configuration

#### `inverse_2d_kinematics(x, y, l0, l1, l2)`
Compute inverse kinematics for a 2D five-bar linkage to find joint angles for a desired end-effector position.

**Parameters:**
- `x, y` (float, float): Desired end-effector position
- `l0` (float): Distance from origin to base pivot points
- `l1` (float): Length of actuated links
- `l2` (float): Length of connecting link

**Returns:**
- `theta1, theta2` (float, float): Joint angles for first solution (radians)
- `theta1_alt, theta2_alt` (float, float): Joint angles for second solution (radians), or None if no solution exists

#### `planar_5_bar_invers(x, y, z, theta_x, theta_y, theta_z, l_0, l_1, l_2, l_3)`
Compute a planar five-bar inverse model for a pose expressed in 3D space, returning the kinematic solution data structure.

**Parameters:**
- `x, y, z` (float): Target position in 3D space
- `theta_x, theta_y, theta_z` (float): Orientation angles (radians)
- `l_0` (float): Base half-distance along x-axis
- `l_1` (float): First link length
- `l_2` (float): Second link length
- `l_3` (float): Offset from the end effector to the planar linkage

**Returns:**
- `plana_5_bar_invers_data`: Structured solution object containing computed joint and end-link angles

### Utility Functions

#### `intersection_points_of_2c(p1, p2, r1, r2)`
Calculate the intersection points of two circles.

**Parameters:**
- `p1, p2` (array-like): Centers of the circles (x, y)
- `r1, r2` (float): Radii of the circles

**Returns:**
- `points` (2x2 array): Two intersection points

## Example Usage

```python
import numpy as np
from five_bar_functions import forward_2d_kinematics, transl, rotz

# Five-bar kinematics
theta1 = np.pi / 4  # 45 degrees
theta2 = np.pi / 3  # 60 degrees
l0 = 1.0
l1 = 1.0
l2 = 1.5

x_1, y_1, x_2, y_2 = forward_2d_kinematics(theta1, theta2, l0, l1, l2)
print(f"Configuration 1: ({x_1:.3f}, {y_1:.3f})")
print(f"Configuration 2: ({x_2:.3f}, {y_2:.3f})")

# Transformation matrices
T = transl(1.0, 2.0, 0.0)
R = rotz(np.pi / 4)
```

## License

MIT
