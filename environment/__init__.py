from os import path

from gym.envs.registration import register


def put_in_model_dir(filename):
    model_dir = 'models'
    return path.join(model_dir, filename)


register(
    id='Particle-v0',
    entry_point='environment.particle:ParticleEnv',
)