#!/usr/bin/env python3

import os
import pickle
import socket
import sys
from copy import deepcopy
from threading import Thread

import mujoco_py
import numpy as np
from gym import spaces
from gym import utils
from gym.envs.mujoco import mujoco_env


class ParticleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'particle.xml', 5)
        utils.EzPickle.__init__(self)
        self._get_viewer()


    def cam_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -90
        self.viewer.cam.distance = 4.90 
        # self.viewer.cam.lookat[1] += 1


    def _step(self, a):
        xposbefore = self.get_body_com("agent")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("agent")[0]

        print(xposbefore, xposafter)
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        reward = 0

        done = not notdone
        ob = self._get_obs()
        return ob, state, reward, done


    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
        ])


    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()


def run():
    env = ParticleEnv()
    env.reset()
    env.cam_setup()

    t = 0

    action = None
    while True:
        env.render()
        obs, r, done, _ = env.step(0)
        if done:
            print('boop!')
            env.reset()
        t += 1

if __name__ == '__main__':
    run()

# while True:
#     sim.data.ctrl[0] =  t * 0.0001
#     sim.data.ctrl[1] =  t* 0.0001
#     t += 1
#     sim.step()
#     viewer.render()
#     print(viewer.cam.distance)
#     if t > 100 and os.getenv('TESTING') is not None:
#         break