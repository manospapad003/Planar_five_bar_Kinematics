#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Planar5BarInverse3DResult:
    a_1: Optional[float] = None
    a_2: Optional[float] = None
    b_1: Optional[float] = None
    b_2: Optional[float] = None
    theta_1: Optional[float] = None
    theta_2: Optional[float] = None
    a_1_2: Optional[float] = None
    a_2_2: Optional[float] = None
    theta_p1: Optional[float] = None
    theta_p2: Optional[float] = None
    theta_end_1: Optional[float] = None
    theta_end_2: Optional[float] = None
    theta_end_n: Optional[float] = None
    theta_mid: Optional[float] = None
    theta_hip: Optional[float] = None


def transl(x,y,z):
    T = np.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T

def rotz(theta):
    R = np.eye(4)
    R[0, 0] = np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[1, 0] = np.sin(theta)
    R[1, 1] = np.cos(theta)
    return R

def roty(theta):
    R = np.eye(4)
    R[0, 0] = np.cos(theta)
    R[0, 2] = np.sin(theta)
    R[2, 0] = -np.sin(theta)
    R[2, 2] = np.cos(theta)
    return R

def rotx(theta):
    R = np.eye(4)
    R[1, 1] = np.cos(theta)
    R[1, 2] = -np.sin(theta)
    R[2, 1] = np.sin(theta)
    R[2, 2] = np.cos(theta)
    return R


def intersection_points_of_2c(p1, p2, r1, r2):
    """
    Returns the intersection points of two circles.

    Parameters:
    p1, p2 : array-like (x, y)
        Centers of the circles
    r1, r2 : float
        Radii of the circles

    Returns:
    points : numpy array (2x2)
        Two intersection points
    """

    p1 = np.array(p1)
    p2 = np.array(p2)

    # Distance between centers
    d = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Check for no solution (optional safety)
    if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
        raise ValueError("No valid intersection points")

    # Intermediate values
    l = (r1**2 - r2**2 + d**2) / (2 * d)
    h_sq = r1**2 - l**2
    if h_sq < 0:
        h_sq = 0.0
    h = np.sqrt(h_sq)

    # Compute intersection points
    x1 = ((l * (p2[0] - p1[0])) / d) + ((h * (p2[1] - p1[1])) / d) + p1[0]
    y1 = ((l * (p2[1] - p1[1])) / d) - ((h * (p2[0] - p1[0])) / d) + p1[1]

    x2 = ((l * (p2[0] - p1[0])) / d) - ((h * (p2[1] - p1[1])) / d) + p1[0]
    y2 = ((l * (p2[1] - p1[1])) / d) + ((h * (p2[0] - p1[0])) / d) + p1[1]

    points = np.array([
        [x1, y1],
        [x2, y2]
    ])

    return points

def forward_2d_kinematics(theta1, theta2, l0, l1, l2):
    """
    Forward kinematics of a planar 5-bar mechanism.

    Parameters:
    theta_a1, theta_a2 : float
        Joint angles (radians)
    l_0 : float
        Base half दूरी
    l_1 : float
        First link length
    l_2 : float
        Second link length

    Returns:
    x_1, y_1, x_2, y_2 : float
        Two possible end-effector positions
    p1, p2 : ndarray
        Homogeneous transforms for the two actuated joints
    """
    ax_start = np.eye(4,4)

    a2 = ax_start @ transl(l0,0,0)
    p2 = a2 @ rotz(theta2) @ transl(l1,0,0)

    a1 = ax_start @ transl(-l0,0,0)
    p1 = a1 @ rotz(theta1) @ transl(l1,0,0)

    d = np.sqrt((p1[0,3] - p2[0,3])**2 + (p1[1,3] - p2[1,3])**2)
    l = (l2**2 - l2**2 + d**2)/(2*d)
    h = np.sqrt(l2**2 - l**2)
    
    x_1 = (l/d)*(p2[0,3] - p1[0,3]) - (h/d)*(p2[1,3] -p1[1,3]) + p1[0,3]
    x_2 = (l/d)*(p2[0,3] - p1[0,3]) + (h/d)*(p2[1,3] -p1[1,3]) + p1[0,3] 
    y_1 = (l/d)*(p2[1,3] - p1[1,3]) + (h/d)*(p2[0,3] -p1[0,3]) + p1[1,3]
    y_2 = (l/d)*(p2[1,3] - p1[1,3]) - (h/d)*(p2[0,3] -p1[0,3]) + p1[1,3]

    return x_1, y_1, x_2, y_2, p1, p2


def inverse_2d_kinematics(x, y, l0, l1, l2):
    """Inverse kinematics for the planar 5-bar mechanism."""
    d1 = np.hypot(x + l0, y)
    d2 = np.hypot(x - l0, y)

    if d1 == 0 or d2 == 0:
        return None, None, None, None

    if d1 > l1 + l2 or d1 < abs(l1 - l2):
        return None, None, None, None
    if d2 > l1 + l2 or d2 < abs(l1 - l2):
        return None, None, None, None

    cos_a1 = np.clip((l1**2 + d1**2 - l2**2) / (2 * l1 * d1), -1.0, 1.0)
    cos_a2 = np.clip((l1**2 + d2**2 - l2**2) / (2 * l1 * d2), -1.0, 1.0)

    a1 = np.arccos(cos_a1)
    a2 = np.arccos(cos_a2)

    b1 = np.arctan2(y, x + l0)
    b2 = np.arctan2(y, x - l0)

    theta1 = b1 + a1
    theta2 = b2 - a2
    theta1_alt = b1 - a1
    theta2_alt = b2 + a2

    return theta1, theta2, theta1_alt, theta2_alt


def forward_3d_kinematics(theta1, theta2, theta3, theta4, theta5, l0, l1, l2, l3):
    """
    Forward kinematics of a 3D five-bar mechanism.

    Parameters:
    theta1, theta2, theta3, theta4, theta5 : float
        Joint angles (radians)
    l0 : float
        Base half distance
    l1 : float
        First link length
    l2 : float
        Second link length
    l3 : float
        End-effector offset from the second link

    Returns:
    x, y, z : float
        End-effector position in 3D space
    p1, p2 : ndarray
        Homogeneous transforms for the two actuated joints
    """
    x_1, y_1, x_2, y_2, p1, p2 = forward_2d_kinematics(theta1, theta2, l0, l1, l2)
    p1 = rotx(theta3) @ p1
    p2 = rotx(theta3) @ p2
    end_effector_5bar = rotx(theta3) @ transl(x_1, y_1, 0)
    end_effector = end_effector_5bar @ rotz(theta4) @ rotx(theta5) @ transl(0, l3, 0)
    return (
        end_effector_5bar[0, 3],
        end_effector_5bar[1, 3],
        end_effector_5bar[2, 3],
        end_effector[0, 3],
        end_effector[1, 3],
        end_effector[2, 3],
        p1,
        p2,
    )

def planar_5_bar_invers_3d(x, y, z, theta_x, theta_y, theta_z, l_0, l_1, l_2, l_3):
    """Inverse kinematics for the 3D five-bar mechanism."""

    # Undo the final offset transform used by forward_3d_kinematics.
    offset_tf = rotx(theta_x) @ rotz(theta_y) @ rotx(theta_z) @ transl(0, l_3, 0)
    offset_vec = np.asarray(offset_tf[:3, 3]).reshape(3)

    x_5b = x - offset_vec[0]
    y_5b = y - offset_vec[1]
    z_5b = z - offset_vec[2]

    # Recover the planar 5-bar coordinates before the x-axis rotation.
    x_planar = x_5b
    if np.isclose(np.cos(theta_x), 0.0):
        y_planar = z_5b / np.sin(theta_x)
    else:
        y_planar = y_5b / np.cos(theta_x)

    theta_a1, theta_a2, theta_a1_alt, theta_a2_alt = inverse_2d_kinematics(
        x_planar, y_planar, l_0, l_1, l_2
    )

    if theta_a1 is None:
        return None

    return Planar5BarInverse3DResult(
        theta_1=theta_a1,
        theta_2=theta_a2,
        theta_end_1=theta_z,
        theta_hip=0,
    )