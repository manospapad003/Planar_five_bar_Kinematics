
"""
Dynamics Example for Five-Bar Kinematics

This script demonstrates how to compute and analyze Jacobian matrices
for five-bar planar mechanisms, including numerical and analytical methods,
singularity detection, and workspace conditioning analysis.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'five_bar_functions'))
import five_bar_functions as fbf


def example_single_jacobian_computation(l0, l1, l2):
    """
    Demonstrate Jacobian computation at a single configuration.
    
    Parameters:
    -----------
    l0 : float
        Base half distance
    l1 : float
        First link length
    l2 : float
        Second link length
    """
    
    print("\n" + "="*70)
    print("Example 1: Single Configuration Jacobian Computation")
    print("="*70)
    
    # Define test angles
    theta1 = np.deg2rad(120)
    theta2 = np.deg2rad(60)
    
    print(f"\nConfiguration:")
    print(f"  theta1 = {np.degrees(theta1):.1f} deg")
    print(f"  theta2 = {np.degrees(theta2):.1f} deg")
    print(f"  l0 = {l0} mm, l1 = {l1} mm, l2 = {l2} mm")
    
    # Compute Jacobian numerically
    print(f"\n--- Numerical Jacobian (delta = 1e-6) ---")
    try:
        J_numerical, x, y = fbf.estimate_jacobian_2d(
            theta1, theta2, l0, l1, l2, delta=1e-6, solution=1
        )
        print(f"End-effector position: ({x:.3f}, {y:.3f}) mm")
        print(f"Jacobian matrix:")
        print(J_numerical)
        det_numerical = fbf.compute_jacobian_determinant(J_numerical)
        print(f"Determinant: {det_numerical:.6f}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Compute Jacobian analytically
    print(f"\n--- Analytical Jacobian ---")
    try:
        J_analytical, x, y = fbf.estimate_jacobian_analytical_2d(
            theta1, theta2, l0, l1, l2, solution=1
        )
        print(f"End-effector position: ({x:.3f}, {y:.3f}) mm")
        print(f"Jacobian matrix:")
        print(J_analytical)
        det_analytical = fbf.compute_jacobian_determinant(J_analytical)
        print(f"Determinant: {det_analytical:.6f}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Compare methods
    if 'J_numerical' in locals() and 'J_analytical' in locals():
        print(f"\n--- Comparison ---")
        error = np.linalg.norm(J_numerical - J_analytical)
        print(f"Frobenius norm of difference: {error:.6e}")
    
    # Compute condition number
    if 'J_analytical' in locals():
        cond = fbf.compute_jacobian_condition_number(J_analytical)
        print(f"Condition number: {cond:.3f}")
        if cond > 100:
            print("  WARNING: Poor conditioning - high sensitivity to input perturbations")
        elif cond > 10:
            print("  WARNING: Moderate conditioning - acceptable for most applications")
        else:
            print("  OK: Good conditioning - low sensitivity to input perturbations")


def example_singularity_detection(l0, l1, l2):
    """
    Demonstrate singularity detection across the workspace.
    
    Parameters:
    -----------
    l0 : float
        Base half distance
    l1 : float
        First link length
    l2 : float
        Second link length
    """
    
    print("\n" + "="*70)
    print("Example 2: Singularity Detection")
    print("="*70)
    
    print(f"\nSearching for singular configurations in the workspace...")
    
    # Scan through angle space
    theta1_range = np.linspace(0, 2 * np.pi, 50)
    theta2_range = np.linspace(0, 2 * np.pi, 50)
    
    singularities = []
    
    for theta1 in theta1_range:
        for theta2 in theta2_range:
            try:
                J, _, _ = fbf.estimate_jacobian_analytical_2d(
                    theta1, theta2, l0, l1, l2, solution=1
                )
                det = fbf.compute_jacobian_determinant(J)
                
                # Identify singularities (determinant near zero)
                if abs(det) < 0.01:
                    singularities.append({
                        'theta1': theta1,
                        'theta2': theta2,
                        'det': det
                    })
            except ValueError:
                # Skip invalid configurations
                continue
    
    if singularities:
        print(f"\nFound {len(singularities)} singular or near-singular configurations:")
        print(f"{'theta1 (deg)':>12} {'theta2 (deg)':>12} {'det(J)':>15}")
        print("-" * 40)
        for sing in singularities[:5]:  # Show first 5
            print(f"{np.degrees(sing['theta1']):>12.1f} {np.degrees(sing['theta2']):>12.1f} {sing['det']:>15.6e}")
        if len(singularities) > 5:
            print(f"... and {len(singularities) - 5} more")
    else:
        print("\nNo singular configurations found in the scanned workspace.")


def example_workspace_jacobian_analysis(l0, l1, l2, resolution=30):
    """
    Compute and visualize Jacobian properties across the workspace.
    
    Parameters:
    -----------
    l0 : float
        Base half distance
    l1 : float
        First link length
    l2 : float
        Second link length
    resolution : int
        Number of samples per motor angle
    """
    
    print("\n" + "="*70)
    print(f"Example 3: Workspace Jacobian Analysis (resolution={resolution}x{resolution})")
    print("="*70)
    
    print(f"\nComputing Jacobian properties across workspace...")
    
    theta1_range = np.linspace(0, 2 * np.pi, resolution)
    theta2_range = np.linspace(0, 2 * np.pi, resolution)
    
    # Storage for results
    end_effector_points = []
    determinants = []
    condition_numbers = []
    
    valid_configs = 0
    
    for theta1 in theta1_range:
        for theta2 in theta2_range:
            try:
                J, x, y = fbf.estimate_jacobian_analytical_2d(
                    theta1, theta2, l0, l1, l2, solution=1
                )
                
                det = fbf.compute_jacobian_determinant(J)
                cond = fbf.compute_jacobian_condition_number(J)
                
                end_effector_points.append([x, y])
                determinants.append(det)
                condition_numbers.append(cond)
                valid_configs += 1
            except ValueError:
                # Skip invalid configurations
                continue
    
    print(f"Valid configurations: {valid_configs}")
    
    end_effector_points = np.array(end_effector_points)
    determinants = np.array(determinants)
    condition_numbers = np.array(condition_numbers)
    
    # Print statistics
    print(f"\n--- Jacobian Determinant Statistics ---")
    print(f"Mean:   {np.mean(determinants):>10.3f}")
    print(f"Std:    {np.std(determinants):>10.3f}")
    print(f"Min:    {np.min(determinants):>10.3f}")
    print(f"Max:    {np.max(determinants):>10.3f}")
    
    print(f"\n--- Condition Number Statistics ---")
    print(f"Mean:   {np.mean(condition_numbers):>10.3f}")
    print(f"Std:    {np.std(condition_numbers):>10.3f}")
    print(f"Min:    {np.min(condition_numbers):>10.3f}")
    print(f"Max:    {np.max(condition_numbers):>10.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: End-effector workspace
    ax = axes[0, 0]
    ax.scatter(end_effector_points[:, 0], end_effector_points[:, 1], 
               c=determinants, cmap='RdYlGn', s=5, alpha=0.6)
    ax.set_xlabel('X (mm)', fontsize=11)
    ax.set_ylabel('Y (mm)', fontsize=11)
    ax.set_title('Workspace colored by Jacobian Determinant', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    cbar1 = plt.colorbar(ax.collections[0], ax=ax)
    cbar1.set_label('det(J)', fontsize=10)
    
    # Plot 2: Workspace colored by condition number
    ax = axes[0, 1]
    scatter = ax.scatter(end_effector_points[:, 0], end_effector_points[:, 1], 
                         c=condition_numbers, cmap='viridis', s=5, alpha=0.6)
    ax.set_xlabel('X (mm)', fontsize=11)
    ax.set_ylabel('Y (mm)', fontsize=11)
    ax.set_title('Workspace colored by Condition Number', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    cbar2 = plt.colorbar(scatter, ax=ax)
    cbar2.set_label('kappa(J)', fontsize=10)
    
    # Plot 3: Determinant histogram
    ax = axes[1, 0]
    ax.hist(determinants, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(determinants), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(determinants):.3f}')
    ax.axvline(0, color='orange', linestyle='--', linewidth=2, label='Singularity')
    ax.set_xlabel('det(J)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Jacobian Determinants', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Condition number histogram
    ax = axes[1, 1]
    ax.hist(condition_numbers, bins=30, color='seagreen', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(condition_numbers), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(condition_numbers):.1f}')
    ax.set_xlabel('kappa(J)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Condition Numbers', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return determinants, condition_numbers


def example_jacobian_velocity_mapping(l0, l1, l2):
    """
    Demonstrate how the Jacobian maps joint velocities to end-effector velocities.
    
    Parameters:
    -----------
    l0 : float
        Base half distance
    l1 : float
        First link length
    l2 : float
        Second link length
    """
    
    print("\n" + "="*70)
    print("Example 4: Jacobian Velocity Mapping")
    print("="*70)
    
    # Configuration
    theta1 = np.deg2rad(100)
    theta2 = np.deg2rad(80)
    
    print(f"\nConfiguration: theta1 = {np.degrees(theta1):.1f} deg, theta2 = {np.degrees(theta2):.1f} deg")
    
    try:
        J, x, y = fbf.estimate_jacobian_analytical_2d(
            theta1, theta2, l0, l1, l2, solution=1
        )
        
        print(f"\nEnd-effector: ({x:.3f}, {y:.3f}) mm")
        print(f"\nJacobian matrix J:")
        print(J)
        
        # Example joint velocities (rad/s)
        joint_velocities = [
            (np.deg2rad(10), 0),        # Only theta1 moving
            (0, np.deg2rad(10)),        # Only theta2 moving
            (np.deg2rad(10), np.deg2rad(5)),  # Both moving
        ]
        
        print(f"\n--- Velocity Mapping Examples ---")
        print(f"End-effector velocity: v_ee = J @ omega_joints")
        print()
        
        for i, (omega1, omega2) in enumerate(joint_velocities, 1):
            omega = np.array([omega1, omega2])
            v_ee = J @ omega
            
            speed = np.linalg.norm(v_ee)
            
            print(f"Case {i}:")
            print(f"  Joint velocities: omega = [{np.degrees(omega1):>6.2f} deg/s, {np.degrees(omega2):>6.2f} deg/s]")
            print(f"  EE velocity:      v = [{v_ee[0]:>8.3f}, {v_ee[1]:>8.3f}] mm/s")
            print(f"  EE speed:              {speed:>8.3f} mm/s")
            print()
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Link parameters (in mm)
    l0 = 31.0         # Base half distance
    l1 = 133.43       # First link length
    l2 = 190.40       # Second link length
    
    print("\n" + "#"*70)
    print("# Five-Bar Dynamics Analysis Examples".center(70))
    print("#"*70)
    print(f"\nLink parameters:")
    print(f"  l0 = {l0} mm (base half distance)")
    print(f"  l1 = {l1} mm (first link)")
    print(f"  l2 = {l2} mm (second link)")
    
    # Run examples
    example_single_jacobian_computation(l0, l1, l2)
    example_singularity_detection(l0, l1, l2)
    example_workspace_jacobian_analysis(l0, l1, l2, resolution=40)
    example_jacobian_velocity_mapping(l0, l1, l2)
    
    print("\n" + "="*70)
    print("All dynamics examples completed!")
    print("="*70 + "\n")
