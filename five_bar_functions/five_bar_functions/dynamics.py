#!/usr/bin/env python3
"""
Dynamics module for five-bar kinematics.

This module provides functions for computing dynamic properties such as
the Jacobian matrix for five-bar planar mechanisms.
"""

import numpy as np
from .kinematics import forward_2d_kinematics


def estimate_jacobian_2d(theta1, theta2, l0, l1, l2, delta=1e-6, solution=1):
    """
    Estimate the Jacobian matrix of a 2D five-bar planar mechanism using numerical differentiation.
    
    The Jacobian matrix relates joint velocities to end-effector velocities:
    [dx/dt]   [J11  J12] [dθ1/dt]
    [dy/dt] = [J21  J22] [dθ2/dt]
    
    Parameters:
    -----------
    theta1 : float
        Joint angle 1 (radians)
    theta2 : float
        Joint angle 2 (radians)
    l0 : float
        Base half distance (mm)
    l1 : float
        First link length (mm)
    l2 : float
        Second link length (mm)
    delta : float
        Step size for numerical differentiation (default 1e-6)
    solution : int
        Which end-effector solution to use (1 or 2, default 1)
    
    Returns:
    --------
    J : ndarray
        2x2 Jacobian matrix
    x : float
        End-effector x position for the given angles
    y : float
        End-effector y position for the given angles
    
    Raises:
    -------
    ValueError
        If forward kinematics fails or invalid solution number provided
    """
    
    if solution not in [1, 2]:
        raise ValueError("solution must be 1 or 2")
    
    # Compute the end-effector position at the nominal angles
    try:
        x_1, y_1, x_2, y_2, _, _ = forward_2d_kinematics(theta1, theta2, l0, l1, l2)
        
        if solution == 1:
            x, y = x_1, y_1
        else:
            x, y = x_2, y_2
    except ValueError as e:
        raise ValueError(f"Forward kinematics failed at θ1={theta1}, θ2={theta2}: {e}")
    
    # Initialize Jacobian matrix
    J = np.zeros((2, 2))
    
    # Compute partial derivatives using finite differences
    # ∂x/∂θ1 and ∂y/∂θ1
    try:
        x_1_p, y_1_p, x_2_p, y_2_p, _, _ = forward_2d_kinematics(
            theta1 + delta, theta2, l0, l1, l2
        )
        if solution == 1:
            x_p, y_p = x_1_p, y_1_p
        else:
            x_p, y_p = x_2_p, y_2_p
        
        J[0, 0] = (x_p - x) / delta  # ∂x/∂θ1
        J[1, 0] = (y_p - y) / delta  # ∂y/∂θ1
    except ValueError:
        # If perturbed configuration is invalid, use zero derivative
        J[0, 0] = 0.0
        J[1, 0] = 0.0
    
    # ∂x/∂θ2 and ∂y/∂θ2
    try:
        x_1_p, y_1_p, x_2_p, y_2_p, _, _ = forward_2d_kinematics(
            theta1, theta2 + delta, l0, l1, l2
        )
        if solution == 1:
            x_p, y_p = x_1_p, y_1_p
        else:
            x_p, y_p = x_2_p, y_2_p
        
        J[0, 1] = (x_p - x) / delta  # ∂x/∂θ2
        J[1, 1] = (y_p - y) / delta  # ∂y/∂θ2
    except ValueError:
        # If perturbed configuration is invalid, use zero derivative
        J[0, 1] = 0.0
        J[1, 1] = 0.0
    
    return J, x, y


def estimate_jacobian_analytical_2d(theta1, theta2, l0, l1, l2, solution=1):
    """
    Compute the Jacobian matrix of a 2D five-bar planar mechanism analytically.
    
    This function derives the Jacobian using the analytical derivatives of the
    forward kinematics equations for the five-bar mechanism.
    
    Parameters:
    -----------
    theta1 : float
        Joint angle 1 (radians)
    theta2 : float
        Joint angle 2 (radians)
    l0 : float
        Base half distance (mm)
    l1 : float
        First link length (mm)
    l2 : float
        Second link length (mm)
    solution : int
        Which end-effector solution to use (1 or 2, default 1)
    
    Returns:
    --------
    J : ndarray
        2x2 Jacobian matrix
    x : float
        End-effector x position for the given angles
    y : float
        End-effector y position for the given angles
    """
    
    if solution not in [1, 2]:
        raise ValueError("solution must be 1 or 2")
    
    # Forward kinematics for the intermediate joints
    # Base positions
    base1 = np.array([-l0, 0])
    base2 = np.array([l0, 0])
    
    # Actuated joint positions
    p1_x = base1[0] + l1 * np.cos(theta1)
    p1_y = base1[1] + l1 * np.sin(theta1)
    p2_x = base2[0] + l1 * np.cos(theta2)
    p2_y = base2[1] + l1 * np.sin(theta2)
    
    # Distance between actuated joints
    d = np.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2)
    
    # Intermediate calculations for end-effector position
    l = (l2**2 - l2**2 + d**2) / (2 * d)  # Note: This simplifies to d/2
    h_sq = l2**2 - l**2
    
    if h_sq < 0:
        raise ValueError("Invalid configuration: no solution exists")
    
    h = np.sqrt(h_sq)
    
    # End-effector positions (both solutions)
    x_1 = (l / d) * (p2_x - p1_x) - (h / d) * (p2_y - p1_y) + p1_x
    y_1 = (l / d) * (p2_y - p1_y) + (h / d) * (p2_x - p1_x) + p1_y
    
    x_2 = (l / d) * (p2_x - p1_x) + (h / d) * (p2_y - p1_y) + p1_x
    y_2 = (l / d) * (p2_y - p1_y) - (h / d) * (p2_x - p1_x) + p1_y
    
    if solution == 1:
        x, y = x_1, y_1
    else:
        x, y = x_2, y_2
    
    # Compute Jacobian elements using analytical derivatives
    # ∂p1/∂θ1
    dp1x_dtheta1 = -l1 * np.sin(theta1)
    dp1y_dtheta1 = l1 * np.cos(theta1)
    
    # ∂p2/∂θ2
    dp2x_dtheta2 = -l1 * np.sin(theta2)
    dp2y_dtheta2 = l1 * np.cos(theta2)
    
    # ∂d/∂θ1
    dd_dtheta1 = ((p2_x - p1_x) * (-dp1x_dtheta1) + (p2_y - p1_y) * (-dp1y_dtheta1)) / d
    
    # ∂d/∂θ2
    dd_dtheta2 = ((p2_x - p1_x) * dp2x_dtheta2 + (p2_y - p1_y) * dp2y_dtheta2) / d
    
    # ∂h/∂d = -d / h (since h = sqrt(l2^2 - (d/2)^2), and l = d/2)
    dh_dd = -d / (2 * h) if h != 0 else 0
    
    # ∂h/∂θ1 and ∂h/∂θ2
    dh_dtheta1 = dh_dd * dd_dtheta1
    dh_dtheta2 = dh_dd * dd_dtheta2
    
    # ∂l/∂d = 1/2 (since l = d/2)
    dl_dd = 0.5
    dl_dtheta1 = dl_dd * dd_dtheta1
    dl_dtheta2 = dl_dd * dd_dtheta2
    
    # Compute Jacobian for solution 1 (solution 2 has inverted sign on h terms)
    sign_h = 1 if solution == 1 else -1
    
    # ∂x/∂θ1
    J11 = (dl_dtheta1 / d) * (p2_x - p1_x) - (l / d**2) * (-dp2x_dtheta2) \
          - (dh_dtheta1 / d) * (p2_y - p1_y) + (h / d**2) * (-dp1y_dtheta1) \
          - sign_h * (l / d) * (-dp1x_dtheta1) - sign_h * (h / d) * (-dp1y_dtheta1) \
          + dp1x_dtheta1
    
    # ∂y/∂θ1
    J21 = (dl_dtheta1 / d) * (p2_y - p1_y) - (l / d**2) * (-dp2y_dtheta2) \
          + (dh_dtheta1 / d) * (p2_x - p1_x) - (h / d**2) * (-dp1x_dtheta1) \
          - sign_h * (l / d) * (-dp1y_dtheta1) + sign_h * (h / d) * (-dp1x_dtheta1) \
          + dp1y_dtheta1
    
    # ∂x/∂θ2
    J12 = (dl_dtheta2 / d) * (p2_x - p1_x) + (l / d**2) * dp2x_dtheta2 \
          - (dh_dtheta2 / d) * (p2_y - p1_y) + (h / d**2) * dp2y_dtheta2
    
    # ∂y/∂θ2
    J22 = (dl_dtheta2 / d) * (p2_y - p1_y) + (l / d**2) * dp2y_dtheta2 \
          + (dh_dtheta2 / d) * (p2_x - p1_x) - (h / d**2) * dp2x_dtheta2
    
    J = np.array([[J11, J12], [J21, J22]])
    
    return J, x, y


def compute_jacobian_determinant(J):
    """
    Compute the determinant of the Jacobian matrix.
    
    The determinant is used to identify singular configurations where
    the mechanism loses degrees of freedom.
    
    Parameters:
    -----------
    J : ndarray
        2x2 Jacobian matrix
    
    Returns:
    --------
    det : float
        Determinant of the Jacobian matrix
    """
    return np.linalg.det(J)


def compute_jacobian_condition_number(J):
    """
    Compute the condition number of the Jacobian matrix.
    
    The condition number indicates the sensitivity of the mechanism to
    input perturbations. Higher values indicate worse conditioning.
    
    Parameters:
    -----------
    J : ndarray
        2x2 Jacobian matrix
    
    Returns:
    --------
    cond : float
        Condition number of the Jacobian matrix
    """
    return np.linalg.cond(J)
