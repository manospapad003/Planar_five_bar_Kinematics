
"""
Workspace Example for Five-Bar Kinematics

This script demonstrates how to use the forward kinematics function to traverse
all possible motor angles and plot the end-effector workspace.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'five_bar_functions'))
import five_bar_functions as fbf


def compute_workspace_2d(l0, l1, l2, resolution=100):
    """
    Compute the 2D workspace by traversing all possible motor angles.
    
    Parameters:
    -----------
    l0 : float
        Base half distance
    l1 : float
        First link length
    l2 : float
        Second link length
    resolution : int
        Number of angle samples per motor (default 100)
    
    Returns:
    --------
    points_1 : ndarray
        All end-effector positions for solution 1 (Nx2 array)
    points_2 : ndarray
        All end-effector positions for solution 2 (Nx2 array)
    """
    
    # Create angle ranges for both motors (0 to 2π)
    theta1_range = np.linspace(0, 2 * np.pi, resolution)
    theta2_range = np.linspace(0, 2 * np.pi, resolution)
    
    points_1 = []
    points_2 = []
    
    # Traverse all combinations of motor angles
    for theta1 in theta1_range:
        for theta2 in theta2_range:
            try:
                # Compute forward kinematics for this angle combination
                x_1, y_1, x_2, y_2, _, _ = fbf.forward_2d_kinematics(
                    theta1, theta2, l0, l1, l2
                )
                points_1.append([x_1, y_1])
                points_2.append([x_2, y_2])
            except ValueError:
                # Skip invalid configurations (no solution)
                continue
    
    return np.array(points_1), np.array(points_2)


def plot_workspace(l0, l1, l2, resolution=100):
    """
    Plot the workspace of the five-bar mechanism.
    
    Parameters:
    -----------
    l0 : float
        Base half distance
    l1 : float
        First link length
    l2 : float
        Second link length
    resolution : int
        Number of angle samples per motor
    """
    
    print("Computing workspace (this may take a moment)...")
    points_1, points_2 = compute_workspace_2d(l0, l1, l2, resolution)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot workspace points for both solutions
    if len(points_1) > 0:
        ax.scatter(points_1[:, 0], points_1[:, 1], c='blue', s=1, alpha=0.5, 
                   label='Solution 1')
    if len(points_2) > 0:
        ax.scatter(points_2[:, 0], points_2[:, 1], c='red', s=1, alpha=0.5, 
                   label='Solution 2')
    
    # Plot base joints
    ax.plot([-l0, l0], [0, 0], 'ko', markersize=8, label='Base joints')
    
    # Set labels and title
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_title('Five-Bar Mechanism Workspace', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()


def example_single_configuration(l0, l1, l2):
    """
    Plot a single configuration showing the linkage geometry.
    
    Parameters:
    -----------
    l0 : float
        Base half distance
    l1 : float
        First link length
    l2 : float
        Second link length
    """
    
    # Choose example angles
    theta1 = np.deg2rad(120)
    theta2 = np.deg2rad(60)
    
    # Compute forward kinematics
    x_1, y_1, x_2, y_2, p1, p2 = fbf.forward_2d_kinematics(theta1, theta2, l0, l1, l2)
    
    # Extract joint positions
    joint1_pos = p1[:2, 3]  # First actuated joint
    joint2_pos = p2[:2, 3]  # Second actuated joint
    base1 = np.array([-l0, 0])
    base2 = np.array([l0, 0])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the linkage
    # First arm (base1 -> joint1 -> end-effector)
    ax.plot([base1[0], joint1_pos[0]], [base1[1], joint1_pos[1]], 'b-', linewidth=2, 
            label='Motor 1 links')
    ax.plot([joint1_pos[0], x_1], [joint1_pos[1], y_1], 'b--', linewidth=2, 
            label='Link 2 (solution 1)')
    
    # Second arm (base2 -> joint2 -> end-effector)
    ax.plot([base2[0], joint2_pos[0]], [base2[1], joint2_pos[1]], 'r-', linewidth=2, 
            label='Motor 2 links')
    ax.plot([joint2_pos[0], x_1], [joint2_pos[1], y_1], 'r--', linewidth=2)
    
    # Plot joints
    ax.plot(*base1, 'ko', markersize=10, label='Base joints')
    ax.plot(*base2, 'ko', markersize=10)
    ax.plot(*joint1_pos, 'bs', markersize=8)
    ax.plot(*joint2_pos, 'rs', markersize=8)
    ax.plot(x_1, y_1, 'g*', markersize=20, label='End-effector (solution 1)')
    ax.plot(x_2, y_2, 'c*', markersize=20, label='End-effector (solution 2)')
    
    # Set labels and title
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_title('Five-Bar Mechanism Configuration\n' + 
                 f'θ₁ = {np.degrees(theta1):.1f}°, θ₂ = {np.degrees(theta2):.1f}°',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Link parameters (in mm)
    l0 = 31.0         # Base half distance
    l1 = 133.43       # First link length
    l2 = 190.40       # Second link length
    
    print("Five-Bar Mechanism Workspace Example")
    print("=" * 50)
    print(f"Link parameters:")
    print(f"  l0 (base half distance): {l0} mm")
    print(f"  l1 (first link): {l1} mm")
    print(f"  l2 (second link): {l2} mm")
    print()
    
    # Show a single configuration first
    print("Plotting single configuration...")
    example_single_configuration(l0, l1, l2)
    
    # Compute and plot the full workspace
    print("\nComputing full workspace...")
    plot_workspace(l0, l1, l2, resolution=100)
