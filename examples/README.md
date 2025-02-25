# PyMuscle Examples

This directory contains examples showing how to use the PyMuscle library.

## Requirements

The examples require additional dependencies beyond the core PyMuscle package. Install them with:

```bash
pip install -r requirements.txt
```

## PyMunk Arm Example

The `pymunk-gym-example.py` file demonstrates a simple arm simulation with a bicep and tricep muscle modeled using PyMuscle and physics simulated with PyMunk.

This example has been updated to work with recent versions of PyMunk (6.x). If you encounter orientation issues, the code has been modified to display the arm with the correct orientation based on the current coordinate system.

### Notes on API changes:

- PyMunk changed its API between versions, moving from `pymunk.constraint.DampedSpring` to `pymunk.DampedSpring`.
- The gravity direction and arm orientation have been adjusted to work with PyMunk 6.x.

## Minimal Physio Example

The `minimal-physio-example.py` shows basic usage of PyMuscle without a physics simulation.

## PyMunk Gym Example  

The `pymunk-gym-example.py` file demonstrates how to use PyMuscle with the OpenAI Gym framework by simulating an arm curl exercise.