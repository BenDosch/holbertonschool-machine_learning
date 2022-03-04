# Apis

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
	1. [Can I join?](#0-can-i-join?)
	2. [Where I am?](#1-where-i-am?)
	3. [Rate me is you can!](#2-rate-me-is-you-can!)
	4. [What will be next?](#3-what-will-be-next?)
	5. [How many by rocket?](#4-how-many-by-rocket?)
4. [Author](#author)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* How to use the Python package requests
* How to make HTTP GET request
* How to handle rate limit
* How to handle pagination
* How to fetch JSON resources
* How to manipulate data from an external service

## Refrences

* [Requests: HTTP for Humans](https://docs.python-requests.org/en/master/ "Requests: HTTP for Humans")

## Tasks
List of tasks with brief descriptions of each task.

### [0. Can I join?](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x01-apis/0-passengers.py "0. Can I join?")

By using the Swapi API, create a method that returns the list of ships that can hold a given number of passengers.

* Prototype: def availableShips(passengerCount):
* Don’t forget the pagination
* If no ship available, return an empty list.

---

### [1. Where I am?](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x01-apis/1-sentience.py "1. Where I am?")

By using the Swapi API, create a method that returns the list of names of the home planets of all sentient species.

* Prototype: def sentientPlanets():
* Don’t forget the pagination

---

### [2. Rate me is you can!](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x01-apis/2-user_location.py "2. Rate me is you can!")

By using the GitHub API, write a script that prints the location of a specific user.

* The user is passed as first argument of the script with the full API URL, example: ./2-user_location.py https://api.github.com/users/holbertonschool
* If the user doesn’t exist, print Not found
* If the status code is 403, print Reset in X min where X is the number of minutes from now and the value of X-Ratelimit-Reset
* Your code should not be executed when the file is imported.

---

### [3. What will be next?](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x01-apis/3-upcoming.py "3. What will be next?")

By using the (unofficial) SpaceX API, write a script that displays the upcoming launch with these information.

* Name of the launch
* The date (in local time)
* The rocket name
* The name (with the locality) of the launchpad
* Format: <launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
* The “upcoming launch” is the one which is the soonest from now, in UTC (we encourage you to use the date_unix for sorting it) - and if 2 launches have the same date, use the first one in the API result.
* Your code should not be executed when the file is imported.

---

### [4. How many by rocket?](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x01-apis/4-rocket_frequency.py "4. How many by rocket?")

By using the (unofficial) SpaceX API, write a script that displays the number of launches per rocket.

* All launches should be taking in consideration
* Each line should contain the rocket name and the number of launches separated by : (format below in the example)
* Order the result by the number launches (descending)
* If multiple rockets have the same amount of launches, order them by alphabetic order (A to Z)
* Your code should not be executed when the file is imported.

---

## Author

[Benjamin Dosch](https://github.com/BenDoschGit)
