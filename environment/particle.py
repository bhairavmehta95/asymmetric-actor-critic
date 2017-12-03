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

t = 0

class ParticleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'particle.xml', 5)
        utils.EzPickle.__init__(self)
        self._get_viewer()


    def _step(self, a):
        xpos = self.get_body_com("agent")[0]
        self.do_simulation(a, self.frame_skip)

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        reward = 0

        done = not notdone
        ob = self._get_obs()

        image_obs = np.zeros([500, 500, 3])

        if self.viewer is not None:
            data = self.viewer.get_image()
            
            img_data = data[0]
            width = data[1]
            height = data[2]
            tmp = np.fromstring(img_data, dtype=np.uint8)
            image_obs = np.reshape(tmp, [height, width, 3])
            image_obs = np.flipud(image_obs)


        return image_obs, state, reward, done


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


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -90
        self.viewer.cam.distance = 4.85 


def run():
    env = ParticleEnv()
    env.reset()

    t = 0

    action = None
    while True:
        obs = env.render()

        # print(width, height)
        # print(obs)
        obs, state, r, done = env.step(np.random.uniform(low=-0.1, high=0.1))
        if done:
            print('boop!')
            env.reset()
        t += 1

if __name__ == '__main__':
    run()
