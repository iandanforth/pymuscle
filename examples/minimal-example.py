from pymuscle import PotvinMuscle as Muscle
from pymuscle.vis import PotvinChart

# Create a Muscle with small number of motor units. Large muscles can have
# thousands of motor units!
motor_unit_count = 120
muscle = Muscle(motor_unit_count)

# Set up the simulation parameters
sim_duration = 60  # seconds
step_size = 1 / 50.0  # 50 frames per second
total_steps = int(sim_duration / step_size)

# Use a constant level of excitation to more easily observe fatigue
excitation = 40.0

total_outputs = []
outputs_by_unit = []
for _ in range(total_steps):
    # Calling step() updates the simulation and returns the total output
    # produced by the muscle during this step for the given excitation level.
    total_output = muscle.step(excitation, step_size)
    total_outputs.append(total_output)
    # You can also introspect the muscle to see the forces being produced
    # by eacy motor unit.
    output_by_unit = muscle.current_forces
    outputs_by_unit.append(output_by_unit)

# Visualize the behavior of the motor units over time
chart = PotvinChart(
    outputs_by_unit,
    step_size
)
chart.display()
