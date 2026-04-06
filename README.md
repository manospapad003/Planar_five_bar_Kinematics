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

## Examples

- `example_kinematics.py` demonstrates 2D forward kinematics and a 3D inverse verification.
- `example_3d_kinematics.py` demonstrates 3D forward kinematics and inverse validation with plotting.

## Package structure

- `five_bar_functions/` — package setup and metadata
- `five_bar_functions/five_bar_functions/kinematics.py` — main kinematics implementation
- `example_kinematics.py` — 2D example script
- `example_3d_kinematics.py` — 3D example script

## License

MIT License
