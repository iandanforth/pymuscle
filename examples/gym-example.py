import gym
from gym.envs.registration import register


register(
    id='PymunkArmEnv-v0',
    entry_point='envs:PymunkArmEnv',
)


def main():
    env = gym.make("PymunkArmEnv-v0")
    env.reset()

    step_size = 1 / 50.0
    sim_duration = 60  # seconds
    total_steps = int(sim_duration / step_size)
    steps = 0
    input_delta = 0.03
    brachialis_input = 0.2
    tricep_input = 0.5
    for _ in range(total_steps):
        steps += 1
        if steps % 50 == 0:
            brachialis_input += input_delta

        if steps % 1000 == 0:
            input_delta *= -1

        env.step(
            [brachialis_input, tricep_input],
            step_size, debug=False
        )
        env.render()


if __name__ == '__main__':
    main()
