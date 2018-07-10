from pymuscle import PotvinMuscle as Muscle
from pymuscle.vis import PotvinChart

# Create a Muscle with small number of motor units.
motor_unit_count = 120
muscle = Muscle(motor_unit_count)

# Set up the simulation parameters
sim_duration = 60  # seconds
frames_per_second = 50
step_size = 1 / frames_per_second
total_steps = int(sim_duration / step_size)

# Use a constant level of excitation to more easily observe fatigue
excitation = 40.0

total_outputs = []
outputs_by_unit = []
print("Starting simulation ...")
for i in range(total_steps):
    # Calling step() updates the simulation and returns the total output
    # produced by the muscle during this step for the given excitation level.
    total_output = muscle.step(excitation, step_size)
    total_outputs.append(total_output)
    # You can also introspect the muscle to see the forces being produced
    # by each motor unit.
    output_by_unit = muscle.current_forces
    outputs_by_unit.append(output_by_unit)
    if (i % (frames_per_second * 10)) == 0:
        print("Sim time - {} seconds ...".format(int(i / frames_per_second)))

# Visualize the behavior of the motor units over time
print("Creating chart ...")
chart = PotvinChart(
    outputs_by_unit,
    step_size
)
# Things to note in the chart:
#   - Some motor units (purple) are never recruited at this level of excitation
#   - Some motor units become completely fatigued in this short time
#   - Some motor units stabilize and decrease their rate of fatigue
#   - Forces from the weakest motor units are almost constant the entire time
chart.display()
