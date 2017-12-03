from environment.particle import ParticleEnv
import numpy as np


def run():
    env = ParticleEnv()
    env.reset()

    tmax = 100
    episodes = 10
    e = 0
    while e < episodes:
        print("new episode")

        goal_obs = env.generate_goal(e)

        t = 0

        while t < tmax:
            obs = env.render()
            state = env.state
            obs, state, r, done = env.step([np.random.uniform(low=-0.5, high=0.5), np.random.uniform(low=-0.5, high=0.5)])

            if done:
                print("Done", r)
                env.reset()

            t += 1

        e += 1

if __name__ == '__main__':
    run()
