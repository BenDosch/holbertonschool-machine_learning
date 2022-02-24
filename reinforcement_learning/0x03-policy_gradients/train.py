#!/usr/bin/env python3
"""Module that """

import numpy as np
import gym
import matplotlib.pyplot as plt
from policy_gradient import policy, policy_gradient

def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Function that implements a full training using the previous function
    created policy_gradient.

    Args:
        env (_type_): The initial gym environment.
        nb_episodes (int): The number of episodes used for training.
        alpha (float, optional): The learning rate. Defaults to 0.000045.
        gamma (float, optional): The discount factor. Defaults to 0.98.
        show_result (bool, optional): When this parameter is True, render the
            environment every 1000 episodes computed. Defaults to False.
    
    Returns: scores
        scores (array[int]): An array where each element is the sum of all
            rewards during one episode loop.
    """
    # Initilaize the weights
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.n
    weights = np.random.rand(num_obs, num_act)  # θ

    # Initilaize scores
    scores = []

    for episode in range(1, nb_episodes + 1):
        # Initilaze values
        state, grads, rewards, score, done  = env.reset(), [], [], 0, False

        # Run episode
        while not done:
            if show_result and episode % 1000 == 0:
                env.render()

            # Get action and grad from policy.
            action, grad = policy_gradient(state[None, :], weights)

            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            next_state = next_state

            # Save gradient and reward for each step
            grads.append(grad)
            rewards.append(reward)

            score += reward

            # Update state, action and grad
            state = next_state

        # Save score and print
        scores.append(score)
        print("Episode: " + str(episode) + " Score: " + str(score) +
              "         ", end="\r", flush=False) 

        # Loop through septs of episode and update weights towards the log
        # policy gradient multiplied by the discounted future rewards.
        # θ ← θ + (α * ∇_θ * E_πθ R(τ))
        # ∇_θ * E_πθ R(τ) =
        # E_πθ(T−1∑t=0(∇_θlogπθ(a_t ∣ s_t)) * T−1∑t′=0(γ^t′-t R(a_t′ ∣ s_t′))
        for i in range(len(grads)):
            weights = weights + (alpha * grads[i] *
                sum([ reward * (gamma ** reward) for reward in rewards[i:]]))
        
    return scores

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    scores = train(env, 10000)
    plt.plot(np.arange(len(scores)), scores)
    plt.show()
    env.close()

    env = gym.make('CartPole-v1')
    scores = train(env, 10000, 0.000045, 0.98, True)
    env.close()
