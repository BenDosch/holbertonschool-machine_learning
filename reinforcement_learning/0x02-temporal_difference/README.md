# Temporal_Difference

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
	1. [Monte Carlo](#0-monte-carlo)
	2. [TD(λ)](#1-td-λ)
	3. [SARSA(λ)](#2-sarsa-λ)

4. [Author](#author)
## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* What is Monte Carlo?
* What is Temporal Difference?
* What is bootstrapping?
* What is n-step temporal difference?
* What is TD(λ)?
* What is an eligibility trace?
* What is SARSA? SARSA(λ)? SARSAMAX?
* What is ‘on-policy’ vs ‘off-policy’?

## Refrences

* [Introduction to reinforcement learning](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ "Introduction to reinforcement learning")
* [Simple Reinforcement Learning: Temporal Difference Learning](https://medium.com/@violante.andre/simple-reinforcement-learning-temporal-difference-learning-e883ea0d65b0 "Simple Reinforcement Learning: Temporal Difference Learning")
* [On-Policy TD Control](https://paperswithcode.com/methods/category/on-policy-td-control "On-Policy TD Control")


## Tasks
List of tasks with brief descriptions of each task.

### [0. Monte Carlo](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/reinforcement_learning/0x02-temporal_difference/0-monte_carlo.py "0. Monte Carlo")

Write a function using the prototpye def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99) that performs the Monte Carlo algorithm. Return a numpy.ndarray with the value estimates of each state.

---

### [1. TD λ](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/reinforcement_learning/0x02-temporal_difference/1-td_lambtha.py "1. TD(λ)")

Write a function using the prototype def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99) that performs the TD(λ) algorithm. Return a numpy.ndarray with the value estimates of each state.

---

### [2. SARSA λ](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/reinforcement_learning/0x02-temporal_difference/2-sarsa_lambtha.py "2. SARSA(λ)")

Write a function using the prototype def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05) that performs the SARSA(λ) algorithm. Return the updated Q table.

---

## Author

[Benjamin Dosch](https://github.com/BenDoschGit)
