from environment.particle import ParticleEnv
import numpy as np


def run(testing=False):
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
            env.reset()
            state = env.state
            obs, state, r, done = env.step([np.random.uniform(low=-0.5, high=0.5), np.random.uniform(low=-0.5, high=0.5)])

            if t == 0 and testing:  
                from scipy import misc
                misc.imsave('goal_obs{}.png'.format(e), goal_obs)
                misc.imsave('first_obs{}.png'.format(e), obs)

            if done:
                print("Done", r)
                env.reset()

            t += 1

        e += 1

if __name__ == '__main__':
    run(testing=False)
