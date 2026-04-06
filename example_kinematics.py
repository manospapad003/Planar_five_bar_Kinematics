import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'five_bar_functions'))

import five_bar_functions as fbf


l0 = 31.0
l1 = 133.43    
l2 = 190.40
# change the angles to test different configurations
theta1 = np.deg2rad(45 + 160) 
theta2 = np.deg2rad(45 + 20)
x_1, y_1, x_2, y_2, p1, p2 = fbf.forward_2d_kinematics(theta1, theta2, l0, l1, l2)

# Compute the joint positions in XY plane
joint1 = p1[:2, 3]
joint2 = p2[:2, 3]
base1 = np.array([-l0, 0])
base2 = np.array([l0, 0])

print(f"End-effector position 1: ({x_1:.3f}, {y_1:.3f})")
# print(f"End-effector position 2: ({x_2:.3f}, {y_2:.3f})")
print(f"Base joint 1: ({base1[0]:.3f}, {base1[1]:.3f})")
# print(f"Base joint 2: ({base2[0]:.3f}, {base2[1]:.3f})")
print(f"Actuated joint 1 position: ({joint1[0]:.3f}, {joint1[1]:.3f})")
# print(f"Actuated joint 2 position: ({joint2[0]:.3f}, {joint2[1]:.3f})")

# Perform inverse kinematics using the planar 5-bar solver for both forward solutions
# for idx, (xe, ye) in enumerate([(x_1, y_1), (x_2, y_2)], start=1):
for idx, (xe, ye) in enumerate([(x_1, y_1)], start=1):
    sol = fbf.planar_5_bar_invers_3d(xe, ye, 0, 0, 0, 0, l0, l1, l2, 0)
    print(sol.theta_1, sol.theta_2, sol.theta_end_1)
    if sol is None:
        print(f"Inverse kinematics did not find a solution for EE {idx}")
        continue

    print(f"Planar 5-bar inverse for EE {idx} at ({xe:.3f}, {ye:.3f}):")
    print(f"  theta_1 = {np.degrees(sol.theta_1):.3f} deg, theta_2 = {np.degrees(sol.theta_2):.3f} deg, "
          f"theta_end_1 = {np.degrees(sol.theta_end_1):.3f} deg")

    x_verify, y_verify, _, _, _, _ = fbf.forward_2d_kinematics(sol.theta_1, sol.theta_2, l0, l1, l2)
    error = np.hypot(x_verify - xe, y_verify - ye)
    print(f"  Forward validation: ({x_verify:.3f}, {y_verify:.3f}), error = {error:.6f}")

# Plot the linkage and the two end-effector solutions
fig, ax = plt.subplots()

# Plot the base pivots, actuated joints, and end-effector solutions
ax.scatter([base1[0], base2[0]], [base1[1], base2[1]], c='k', marker='o', label='Base pivots')
ax.scatter([joint1[0], joint2[0]], [joint1[1], joint2[1]], c='tab:blue', marker='s', label='Actuated joints')
ax.scatter([x_1], [y_1], c='tab:red', marker='X', s=100, label='End effector')

# Draw linkage lines for each solution
ax.plot([base1[0], joint1[0]], [base1[1], joint1[1]], 'tab:blue', linestyle='--', label='Link 1')
ax.plot([base2[0], joint2[0]], [base2[1], joint2[1]], 'tab:orange', linestyle='--', label='Link 2')
ax.plot([joint1[0], x_1], [joint1[1], y_1], 'tab:red', linestyle='-.', alpha=0.7)
ax.plot([joint2[0], x_1], [joint2[1], y_1], 'tab:red', linestyle='-.', alpha=0.7)
# ax.plot([joint1[0], x_2], [joint1[1], y_2], 'tab:green', linestyle='-.', alpha=0.7)
# ax.plot([joint2[0], x_2], [joint2[1], y_2], 'tab:green', linestyle='-.', alpha=0.7)

ax.text(x_1, y_1, '  EE 1', color='tab:red')
# ax.text(x_2, y_2, '  EE 2', color='tab:green')
ax.text(joint1[0], joint1[1], '  J1', color='tab:blue')
ax.text(joint2[0], joint2[1], '  J2', color='tab:orange')

ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Five-Bar Linkage Forward Kinematics')
ax.legend(loc='best')
ax.grid(True)

plt.show()



