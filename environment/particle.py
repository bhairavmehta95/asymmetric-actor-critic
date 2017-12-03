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
    def __init__(self, dist_epsilon=0.1):
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.getcwd(), 'environment/assets/particle.xml'), 5)
        utils.EzPickle.__init__(self)
        self.dist_epsilon = dist_epsilon

        self.state = None
        self.state_dim = self.state_vector().shape
        high = np.inf*np.ones(self.state_dim)
        low = -high
        self.state_space = spaces.Box(low, high)


    def _step(self, a):
        xpos = self.get_body_com("agent")[0]
        self.do_simulation(a, self.frame_skip)
        self.state = state = self.state_vector()

        notdone = np.isfinite(state).all() \
            and not self._at_goal() and not self._escaped()

        reward = 0

        if self._at_goal():
            reward = 1

        done = not notdone

        image_obs = np.zeros([500, 500, 3])
        if self.viewer is not None:
            image_obs = self._get_image()

        return image_obs, state, reward, done


    def _get_image(self):
        data = self._get_viewer().get_image()

        img_data = data[0]
        width = data[1]
        height = data[2]

        tmp = np.fromstring(img_data, dtype=np.uint8)
        image_obs = np.reshape(tmp, [height, width, 3])
        image_obs = np.flipud(image_obs)

        return image_obs


    def _get_state(self, body='agent'):
        state = self.get_body_com(body)[:2]
        return state


    def _at_goal(self):
        try:
            self.goal_state
        except:
            return False

        return np.linalg.norm(self._get_state() - self.goal_state) \
            < self.dist_epsilon


    def _escaped(self):
        state = self._get_state()
        return state.any() > 2


    def _set_goal(self, goal):
        qpos = goal['qpos']
        qvel = goal['qvel']

        self.goal_state = qpos
        self.set_state(qpos, qvel)
        self.step(qvel)

        goal_obs = self._get_image()

        return goal_obs


    def generate_goal(self, e):
        goalposx = np.random.uniform(low=-2.0, high=2.0)
        goalposy = np.random.uniform(low=-2.0, high=2.0)

        goalpos = np.array([goalposx, goalposy])
        goalvel = np.zeros(shape=self.model.nq)

        goal = dict()
        goal['qpos'] = goalpos
        goal['qvel'] = goalvel

        goal_obs = self._set_goal(goal)

        self.reset()

        return goal_obs


    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_image()


    def viewer_setup(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(init_width=100, init_height=100)
            self.viewer.start()
            self.viewer.set_model(self.model)

        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -90
        self.viewer.cam.distance = 4.85 
        self.viewer.cam.height = 100
        self.viewer.cam.width = 100

