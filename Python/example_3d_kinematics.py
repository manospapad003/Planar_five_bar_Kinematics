import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'five_bar_functions'))

import five_bar_functions as fbf

# Parameters for the 3D five-bar mechanism
l0 = 31.0
l1 = 133.43
l2 = 190.40
l3 = 10.0  # End-effector offset

# Joint angles (radians)
theta1 = np.deg2rad(45+90)
theta2 = np.deg2rad(45)
theta3 = np.deg2rad(30)  # Rotation around y-axis
theta4 = np.deg2rad(0)  # Rotation around x-axis
theta5 = np.deg2rad(0)  # Rotation around z-axis (end-effector orientation)
# Compute forward 3D kinematics
x_5bar, y_5bar, z_5bar, x, y, z, p1, p2 = fbf.forward_3d_kinematics(theta1, theta2, theta3, theta4, theta5, l0, l1, l2, l3)

print(f"Five-bar end-effector position: ({x_5bar:.3f}, {y_5bar:.3f}, {z_5bar:.3f})")
print(f"Final end-effector position: ({x:.3f}, {y:.3f}, {z:.3f})")
print(f"Actuated joint 1 position: ({p1[0, 3]:.3f}, {p1[1, 3]:.3f}, {p1[2, 3]:.3f})")
print(f"Actuated joint 2 position: ({p2[0, 3]:.3f}, {p2[1, 3]:.3f}, {p2[2, 3]:.3f})")

# Optionally, print the full transforms
# print("\nJoint 1 transform:")
# print(p1)
# print("\nJoint 2 transform:")
# print(p2)

# Plot the 3D linkage
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Base points
base1 = np.array([-l0, 0, 0])
base2 = np.array([l0, 0, 0])

# Actuated joint positions
joint1 = p1[:3, 3]
joint2 = p2[:3, 3]

# Five-bar end-effector position
ee_5bar = np.array([x_5bar, y_5bar, z_5bar])

# Final end-effector position
end_effector = np.array([x, y, z])

# Plot points
ax.scatter([base1[0], base2[0]], [base1[1], base2[1]], [base1[2], base2[2]], c='k', marker='o', s=100, label='Base pivots')
ax.scatter([joint1[0], joint2[0]], [joint1[1], joint2[1]], [joint1[2], joint2[2]], c='tab:blue', marker='s', s=100, label='Actuated joints')
ax.scatter([ee_5bar[0]], [ee_5bar[1]], [ee_5bar[2]], c='tab:purple', marker='D', s=100, label='5-bar EE')
ax.scatter([end_effector[0]], [end_effector[1]], [end_effector[2]], c='tab:red', marker='X', s=200, label='Final EE')

# Plot links
ax.plot([base1[0], joint1[0]], [base1[1], joint1[1]], [base1[2], joint1[2]], 'tab:blue', linewidth=3, label='Link 1')
ax.plot([base2[0], joint2[0]], [base2[1], joint2[1]], [base2[2], joint2[2]], 'tab:orange', linewidth=3, label='Link 2')
ax.plot([joint1[0], ee_5bar[0]], [joint1[1], ee_5bar[1]], [joint1[2], ee_5bar[2]], 'tab:purple', linewidth=2, linestyle='--', label='Connector 1')
ax.plot([joint2[0], ee_5bar[0]], [joint2[1], ee_5bar[1]], [joint2[2], ee_5bar[2]], 'tab:purple', linewidth=2, linestyle='--', label='Connector 2')
ax.plot([ee_5bar[0], end_effector[0]], [ee_5bar[1], end_effector[1]], [ee_5bar[2], end_effector[2]], 'tab:red', linewidth=2, linestyle='-', label='EE Offset')

# Labels
ax.text(base1[0], base1[1], base1[2], '  Base 1', color='k')
ax.text(base2[0], base2[1], base2[2], '  Base 2', color='k')
ax.text(joint1[0], joint1[1], joint1[2], '  J1', color='tab:blue')
ax.text(joint2[0], joint2[1], joint2[2], '  J2', color='tab:orange')
ax.text(ee_5bar[0], ee_5bar[1], ee_5bar[2], '  5-bar EE', color='tab:purple')
ax.text(end_effector[0], end_effector[1], end_effector[2], '  Final EE', color='tab:red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Five-Bar Linkage Forward Kinematics')
ax.legend()
ax.grid(True)

# Set equal aspect ratio for 3D plot
all_points = np.array([base1, base2, joint1, joint2, ee_5bar, end_effector])
max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                      all_points[:, 1].max() - all_points[:, 1].min(),
                      all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0

mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()

# Inverse kinematics example
print("\n" + "="*60)
print("Inverse Kinematics Example")
print("="*60)

# Use the final end-effector position from forward kinematics as input
inv_sol = fbf.planar_5_bar_invers_3d(x, y, z, theta3, theta4, theta5, l0, l1, l2, l3)

print(f"\nInverse kinematics for target position: ({x:.3f}, {y:.3f}, {z:.3f})")
print(f"  theta_hip = {np.degrees(inv_sol.theta_hip):.3f} deg")
print(f"  theta_1 = {np.degrees(inv_sol.theta_1):.3f} deg")
print(f"  theta_2 = {np.degrees(inv_sol.theta_2):.3f} deg")
print(f"  theta_end_1 = {np.degrees(inv_sol.theta_end_1):.3f} deg")

# Verify by computing forward kinematics with the inverse solution
x_verify_5bar, y_verify_5bar, z_verify_5bar, x_verify, y_verify, z_verify, _, _ = \
    fbf.forward_3d_kinematics(inv_sol.theta_1, inv_sol.theta_2, theta3, theta4, theta5, l0, l1, l2, l3)

print(f"\nForward verification from inverse solution:")
print(f"  Original 5-bar EE: ({x_5bar:.3f}, {y_5bar:.3f}, {z_5bar:.3f})")
print(f"  Verified 5-bar EE: ({x_verify_5bar:.3f}, {y_verify_5bar:.3f}, {z_verify_5bar:.3f})")
print(f"  Original final EE: ({x:.3f}, {y:.3f}, {z:.3f})")
print(f"  Verified final EE: ({x_verify:.3f}, {y_verify:.3f}, {z_verify:.3f})")

position_error = np.sqrt((x_verify - x)**2 + (y_verify - y)**2 + (z_verify - z)**2)
print(f"  Final EE position error: {position_error:.6f}")

# Compare inverse solution angles with original input angles
print(f"\nAngle Comparison:")
print(f"  Original theta1: {np.degrees(theta1):.3f} deg, Inverse theta1: {np.degrees(inv_sol.theta_1):.3f} deg")
print(f"  Original theta2: {np.degrees(theta2):.3f} deg, Inverse theta2: {np.degrees(inv_sol.theta_2):.3f} deg")