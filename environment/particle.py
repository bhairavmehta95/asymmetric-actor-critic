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
from scipy import misc

t = 0

class ParticleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'particle.xml', 5)
        utils.EzPickle.__init__(self)
        self.state = None


    def _step(self, a):
        xpos = self.get_body_com("agent")[0]
        self.do_simulation(a, self.frame_skip)
        self.state = state = self.state_vector()

        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        reward = 0

        done = not notdone

        image_obs = np.zeros([500, 500, 3])

        if self.viewer is not None:
            image_obs = self._get_image()

        return image_obs, state, reward, done


    def _get_image(self):
        self.render()
        data = self.viewer.get_image()
        
        img_data = data[0]
        width = data[1]
        height = data[2]
        tmp = np.fromstring(img_data, dtype=np.uint8)
        image_obs = np.reshape(tmp, [height, width, 3])
        image_obs = np.flipud(image_obs)

        return image_obs



    def _at_goal(self):
        pass


    def _escaped(self):
        pass


    def _get_goal(self, goal):
        qpos = goal['qpos']
        qvel = goal['qvel']

        self.set_state(qpos, qvel)
        goal_obs = self._get_image()
        return goal_obs


    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_image()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -90
        self.viewer.cam.distance = 4.85 


def run():
    env = ParticleEnv()
    env.reset()

    goalpos = np.random.uniform(size=env.model.nq, low=1.1, high=1.4)
    goalvel = np.zeros(shape=env.model.nq)

    goal = dict()
    goal['qpos'] = goalpos
    goal['qvel'] = goalvel
    goal_obs = env._get_goal(goal)

    env.reset()

    t = 0

    action = None
    while True:
        obs = env.render()

        # print(width, height)
        # print(obs)

        obs, state, r, done = env.step(np.random.uniform(low=-0.1, high=0.1))
        if done:
            # print('boop!')
            env.reset()
        t += 1

if __name__ == '__main__':
    run()
