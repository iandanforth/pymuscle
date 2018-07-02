import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
from pymuscle import PotvinMuscle # noqa

muscle = PotvinMuscle(120, True)

# Performance Benchmarking
start = time.time()
iterations = 10000
step_size = 1 / 50.0
for _ in range(iterations):
    excitation = np.random.random_integers(1, 60) / 1.0  # Quick cast
    force = muscle.step(excitation, step_size)
duration = time.time() - start
avg = duration / iterations

multiple = 100
real = 1.0 / 60.0
x_real = real / multiple

print("{} iterations took {} seconds. {} per iteration".format(
    iterations, duration, avg)
)

if avg < x_real:
    print("This is better than {}x real time :)".format(multiple))
else:
    print("This is worse than {}x real time. :(".format(multiple))

print("Multiple:", real / avg)
