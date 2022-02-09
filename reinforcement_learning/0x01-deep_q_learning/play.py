#!/usr/bin/env python3
"""Module that contains """

import time
import gym
from tensorflow.keras.optimizers import Adam
PreprocessAtari = __import__('train').PreprocessAtari
build_model =  __import__('train').build_model
build_agent =  __import__('train').build_agent

def play():
    # Setup envirionment
    env = gym.make("Breakout-v0")
    env = PreprocessAtari(env)
    
    # Set up model
    height, width, channels = env.observation_space.shape
    actions = env.action_space.n
    # env.unwrapped.get_action_meanings()
    model = build_model(height, width, channels, actions)
    # model.summary()

    # Set up agent
    dqn = build_agent(model, actions)
    dqn.compile(Adam(learning_rate=1e-4), metrics=['mae'])

    # Load weights
    dqn.load_weights('./policy.h5')

    # Play
    dqn.test(env, nb_episodes=10, visualize=True)


if __name__ == '__main__':
    play()
