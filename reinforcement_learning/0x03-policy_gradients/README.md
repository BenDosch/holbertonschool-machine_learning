# Policy_Gradients

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
	1. [Simple Policy function](#0-simple-policy-function)
	2. [Compute the Monte-Carlo policy gradient](#1-compute-the-monte-carlo-policy-gradient)
	3. [Implement the training](#2-implement-the-training)
	4. [Animate iteration](#3-animate-iteration)

4. [Author](#author)
## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* What is Policy?
* How to calculate a Policy Gradient?
* What and how to use a Monte-Carlo policy gradient?

## Refrences

* [How Policy Gradient Reinforcement Learning Works](https://www.youtube.com/watch?v=A_2U6Sx67sE "How Policy Gradient Reinforcement Learning Works")
* [Policy Gradients in a Nutshell](https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d "Policy Gradients in a Nutshell")
* [Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html "Policy Gradient Algorithms")
* [RL Course by David Silver - Lecture 7: Policy Gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs "RL Course by David Silver - Lecture 7: Policy Gradient Methods")

## Tasks
List of tasks with brief descriptions of each task.

### [0. Simple Policy function](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/reinforcement_learning/0x03-policy_gradients/policy_gradient.py "0. Simple Policy function")

Write a function that computes to policy with a weight of a matrix from the prototype: def policy(matrix, weight).

---

### [1. Compute the Monte-Carlo policy gradient](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/reinforcement_learning/0x03-policy_gradients/policy_gradient.py "1. Compute the Monte-Carlo policy gradient")

By using the previous function created policy, write a function that computes the Monte-Carlo policy gradient based on a state and a weight matrix with the prototype: def policy_gradient(state, weight). state is matrix representing the current observation of the environment and weight is matrix of random weight. Return the action and the gradient.

---

### [2. Implement the training](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/reinforcement_learning/0x03-policy_gradients/train.py "2. Implement the training")

By using the previous function created policy_gradient, write a function that implements a full training from the prototype: def train(env, nb_episodes, alpha=0.000045, gamma=0.98). env is initial environment, nb_episodes is number of episodes used for training, alpha is the learning rate, and gamma is the discount factor.
Return all values of the score, the sum of all rewards during one episode loop.

---

### [3. Animate iteration](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/reinforcement_learning/0x03-policy_gradients/train.py "3. Animate iteration")

Update the prototype of the train function by adding a last optional parameter show_result (default: False). When this parameter is True, render the environment every 1000 episodes computed.

---

## Author

[Benjamin Dosch](https://github.com/BenDoschGit)
