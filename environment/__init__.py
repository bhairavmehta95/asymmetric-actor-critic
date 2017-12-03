from os import path

from gym.envs.registration import register

register(
    id='Particle-v0',
    entry_point='environment.particle:ParticleEnv',
)