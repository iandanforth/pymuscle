import numpy as np
from gym import utils
from . import mujoco_env
from pymuscle import PotvinFuglevandMuscle as Muscle


class MuscledHopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, apply_fatigue=False):
        # Instantiate the PyMuscles (not real muscle names)
        hamstring_motor_unit_count = 300
        self.hamstring_muscle = Muscle(hamstring_motor_unit_count, apply_fatigue)
        thigh_motor_unit_count = 300
        self.thigh_muscle = Muscle(thigh_motor_unit_count, apply_fatigue)
        calf_motor_unit_count = 200
        self.calf_muscle = Muscle(calf_motor_unit_count, apply_fatigue)
        shin_motor_unit_count = 100
        self.shin_muscle = Muscle(shin_motor_unit_count, apply_fatigue)
        self.muscles = [
            self.hamstring_muscle,
            self.thigh_muscle,
            self.calf_muscle,
            self.shin_muscle
        ]

        # Initialize parents
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]

        # Get output from muscles
        outputs = [muscle.step(a[i], 0.002 * self.frame_skip) for
                   i, muscle in enumerate(self.muscles)]

        self.do_simulation(outputs, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-.005,
            high=.005,
            size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-.005,
            high=.005,
            size=self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
