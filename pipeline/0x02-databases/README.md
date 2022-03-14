# Databases

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
	1. [Create a database](#0-create-a-database)
	2. [First table](#1-first-table)
	3. [List all in table](#2-list-all-in-table)
	4. [First add](#3-first-add)
	5. [Select the best](#4-select-the-best)
	6. [Average](#5-average)
	7. [Temperatures #0](#6-temperatures-#0)
	8. [Temperatures #2](#7-temperatures-#2)
	9. [Genre ID by show](#8-genre-id-by-show)
	10. [No genre](#9-no-genre)
	11. [Number of shows by genre](#10-number-of-shows-by-genre)
	12. [Rotten tomatoes](#11-rotten-tomatoes)
	13. [Best genre](#12-best-genre)
	14. [We are all unique!](#13-we-are-all-unique!)
	15. [In and not out](#14-in-and-not-out)
	16. [Best band ever!](#15-best-band-ever!)
	17. [Old school band](#16-old-school-band)
	18. [Buy buy buy](#17-buy-buy-buy)
	19. [Email validation to sent](#18-email-validation-to-sent)
	20. [Add bonus](#19-add-bonus)
	21. [Average score](#20-average-score)
	22. [Safe divide](#21-safe-divide)
	23. [List all databases](#22-list-all-databases)
	24. [Create a database](#23-create-a-database)
	25. [Insert document](#24-insert-document)
	26. [All documents](#25-all-documents)
	27. [All matches](#26-all-matches)
	28. [Count](#27-count)
	29. [Update](#28-update)
	30. [Delete by match](#29-delete-by-match)
	31. [List all documents in Python](#30-list-all-documents-in-python)
	32. [Insert a document in Python](#31-insert-a-document-in-python)
	33. [Change school topics](#32-change-school-topics)
	34. [Where can I learn Python?](#33-where-can-i-learn-python?)
	35. [Log stats](#34-log-stats)
4. [Advanced Tasks](#advanced-tasks)
	36. [Optimize simple search](#35-optimize-simple-search)
	37. [Optimize search and score](#36-optimize-search-and-score)
	38. [No table for a meeting](#37-no-table-for-a-meeting)
	39. [Average weighted score](#38-average-weighted-score)
	40. [Regex filter](#39-regex-filter)
	41. [Top students](#40-top-students)
	42. [Log stats - new version](#41-log-stats---new-version)
5. [Author](#author)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* What’s a relational database
* What’s a none relational database
* What is difference between SQL and NoSQL
* How to create tables with constraints
* How to optimize queries by adding indexes
* What is and how to implement stored procedures and functions in MySQL
* What is and how to implement views in MySQL
* What is and how to implement triggers in MySQL
* What is ACID
* What is a document storage
* What are NoSQL types
* What are benefits of a NoSQL database
* How to query information from a NoSQL database
* How to insert/update/delete information from a NoSQL database
* How to use MongoDB

## Refrences

* [Title](www.url.com "Title")

## Tasks
List of tasks with brief descriptions of each task.

### [0. Create a database](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/0-create_database_if_missing.sql "0. Create a database")

Write a script that creates the database db_0 in your MySQL server.

* If the database db_0 already exists, your script should not fail
* You are not allowed to use the SELECT or SHOW statements

---

### [1. First table](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/1-first_table.sql "1. First table")

Write a script that creates a table called first_table in the current database in your MySQL server.

* first_table description:
* id INT
* name VARCHAR(256)
* The database name will be passed as an argument of the mysql command
* If the table first_table already exists, your script should not fail
* You are not allowed to use the SELECT or SHOW statements

---

### [2. List all in table](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/2-list_values.sql "2. List all in table")

Write a script that lists all rows of the table first_table in your MySQL server.

* All fields should be printed
* The database name will be passed as an argument of the mysql command

---

### [3. First add](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/3-insert_value.sql "3. First add")

Write a script that inserts a new row in the table first_table in your MySQL server.

* New row:
* id = 89
* name = Holberton School
* The database name will be passed as an argument of the mysql command

---

### [4. Select the best](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/4-best_score.sql "4. Select the best")

Write a script that lists all records with a score >= 10 in the table second_table in your MySQL server.

* Results should display both the score and the name (in this order)
* Records should be ordered by score (top first)
* The database name will be passed as an argument of the mysql command

---

### [5. Average](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/5-average.sql "5. Average")

Write a script that computes the score average of all records in the table second_table in your MySQL server.

* The result column name should be average
* The database name will be passed as an argument of the mysql command

---

### [6. Temperatures #0](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/6-avg_temperatures.sql "6. Temperatures #0")

Import in hbtn_0c_0 database this table dump: [download](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-higher-level_programming+/272/temperatures.sql)

Write a script that displays the average temperature (Fahrenheit) by city ordered by temperature (descending).

---

### [7. Temperatures #2](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/7-max_state.sql "7. Temperatures #2")

Import in hbtn_0c_0 database this table dump: download (same as Temperatures #0)

Write a script that displays the max temperature of each state (ordered by State name).

---

### [8. Genre ID by show](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/8-genre_id_by_show.sql "8. Genre ID by show")

Import the database dump from hbtn_0d_tvshows to your MySQL server: [download](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-higher-level_programming+/274/hbtn_0d_tvshows.sql)

Write a script that lists all shows contained in hbtn_0d_tvshows that have at least one genre linked.

* Each record should display: tv_shows.title - tv_show_genres.genre_id
* Results must be sorted in ascending order by tv_shows.title and tv_show_genres.genre_id
* You can use only one SELECT statement
* The database name will be passed as an argument of the mysql command

---

### [9. No genre](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/9-no_genre.sql "9. No genre")

Import the database dump from hbtn_0d_tvshows to your MySQL server: [download](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-higher-level_programming+/274/hbtn_0d_tvshows.sql)

Write a script that lists all shows contained in hbtn_0d_tvshows without a genre linked.

* Each record should display: tv_shows.title - tv_show_genres.genre_id
* Results must be sorted in ascending order by tv_shows.title and tv_show_genres.genre_id
* You can use only one SELECT statement
* The database name will be passed as an argument of the mysql command

---

### [10. Number of shows by genre](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/10-count_shows_by_genre.sql "10. Number of shows by genre")

Import the database dump from hbtn_0d_tvshows to your MySQL server: [download](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-higher-level_programming+/274/hbtn_0d_tvshows.sql)

Write a script that lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.

* Each record should display: <TV Show genre> - <Number of shows linked to this genre>
* First column must be called genre
* Second column must be called number_of_shows
* Don’t display a genre that doesn’t have any shows linked
* Results must be sorted in descending order by the number of shows linked
* You can use only one SELECT statement
* The database name will be passed as an argument of the mysql command

---

### [11. Rotten tomatoes](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/11-rating_shows.sql "11. Rotten tomatoes")

Import the database hbtn_0d_tvshows_rate dump to your MySQL server: [download](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-higher-level_programming+/274/hbtn_0d_tvshows_rate.sql)

Write a script that lists all shows from hbtn_0d_tvshows_rate by their rating.

* Each record should display: tv_shows.title - rating sum
* Results must be sorted in descending order by the rating
* You can use only one SELECT statement
* The database name will be passed as an argument of the mysql command

---

### [12. Best genre](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/12-rating_genres.sql "12. Best genre")

Import the database dump from hbtn_0d_tvshows_rate to your MySQL server: [download](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-higher-level_programming+/274/hbtn_0d_tvshows_rate.sql)

Write a script that lists all genres in the database hbtn_0d_tvshows_rate by their rating.

* Each record should display: tv_genres.name - rating sum
* Results must be sorted in descending order by their rating
* You can use only one SELECT statement
* The database name will be passed as an argument of the mysql command

---

### [13. We are all unique!](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/13-uniq_users.sql "13. We are all unique!")

Write a SQL script that creates a table users following these requirements:

* With these attributes:
	* id, integer, never null, auto increment and primary key
	* email, string (255 characters), never null and unique
	* name, string (255 characters)
* If the table already exists, your script should not fail
* Your script can be executed on any database

Context: Make an attribute unique directly in the table schema will enforced your business rules and avoid bugs in your application

---

### [14. In and not out](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/14-country_users.sql "14. In and not out")

Write a SQL script that creates a table users following these requirements:

* With these attributes:
	* id, integer, never null, auto increment and primary key
	* email, string (255 characters), never null and unique
	* name, string (255 characters)
	* country, enumeration of countries: US, CO and TN, never null (= default will be the first element of the enumeration, here US)
* If the table already exists, your script should not fail
* Your script can be executed on any database

---

### [15. Best band ever!](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/15-fans.sql "15. Best band ever!")

Write a SQL script that ranks country origins of bands, ordered by the number of (non-unique) fans

Requirements:

* Import this table dump: metal_bands.sql.zip
* Column names must be: origin and nb_fans
* Your script can be executed on any database

Context: Calculate/compute something is always power intensive… better to distribute the load!

---

### [16. Old school band](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/16-glam_rock.sql "16. Old school band")

Write a SQL script that lists all bands with Glam rock as their main style, ranked by their longevity

Requirements:

* Import this table dump: metal_bands.sql.zip
* Column names must be:
* band_name
* lifespan until 2020 (in years)
* You should use attributes formed and split for computing the lifespan
* Your script can be executed on any database

---

### [17. Buy buy buy](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/17-store.sql "17. Buy buy buy")

Write a SQL script that creates a trigger that decreases the quantity of an item after adding a new order.

Quantity in the table items can be negative.

Context: Updating multiple tables for one action from your application can generate issue: network disconnection, crash, etc… to keep your data in a good shape, let MySQL do it for you!

---

### [18. Email validation to sent](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/18-valid_email.sql "18. Email validation to sent")

Write a SQL script that creates a trigger that resets the attribute valid_email only when the email has been changed.

Context: Nothing related to MySQL, but perfect for user email validation - distribute the logic to the database itself!

---

### [19. Add bonus](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/19-bonus.sql "19. Add bonus")

Write a SQL script that creates a stored procedure AddBonus that adds a new correction for a student.

Requirements:

* Procedure AddBonus is taking 3 inputs (in this order):
	* user_id, a users.id value (you can assume user_id is linked to an existing users)
	* project_name, a new or already exists projects - if no projects.name found in the table, you should create it
	* score, the score value for the correction

Context: Write code in SQL is a nice level up!

---

### [20. Average score](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/20-average_score.sql "20. Average score")

Write a SQL script that creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.

Requirements:

* Procedure ComputeAverageScoreForUser is taking 1 input:
	* user_id, a users.id value (you can assume user_id is linked to an existing users)

---

### [21. Safe divide](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/21-div.sql "21. Safe divide")

Write a SQL script that creates a function SafeDiv that divides (and returns) the first by the second number or returns 0 if the second number is equal to 0.

Requirements:

* You must create a function
* The function SafeDiv takes 2 arguments:
	* a, INT
	* b, INT
* And returns a / b or 0 if b == 0

---

### [22. List all databases](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/22-list_databases "22. List all databases")

Write a script that lists all databases in MongoDB.

---

### [23. Create a database](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/23-use_or_create_database "23. Create a database")

Write a script that creates or uses the database my_db:

---

### [24. Insert document](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/24-insert "24. Insert document")

Write a script that inserts a document in the collection school:

* The document must have one attribute name with value “Holberton school”
* The database name will be passed as option of mongo command

---

### [25. All documents](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/25-all "25. All documents")

Write a script that lists all documents in the collection school:

* The database name will be passed as option of mongo command

---

### [26. All matches](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/26-match "26. All matches")

Write a script that lists all documents with name="Holberton school" in the collection school:

* The database name will be passed as option of mongo command

---

### [27. Count](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/27-count "27. Count")

Write a script that displays the number of documents in the collection school:

* The database name will be passed as option of mongo command

---

### [28. Update](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/28-update "28. Update")

Write a script that adds a new attribute to a document in the collection school:

* The script should update only document with name="Holberton school" (all of them)
* The update should add the attribute address with the value “972 Mission street”
* The database name will be passed as option of mongo command

---

### [29. Delete by match](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/29-delete "29. Delete by match")

Write a script that deletes all documents with name="Holberton school" in the collection school:

* The database name will be passed as option of mongo command

---

### [30. List all documents in Python](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/30-all.py "30. List all documents in Python")

Write a Python function that lists all documents in a collection:

* Prototype: def list_all(mongo_collection):
* Return an empty list if no document in the collection
* mongo_collection will be the pymongo collection object

---

### [31. Insert a document in Python](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/31-insert_school.py "31. Insert a document in Python")

Write a Python function that inserts a new document in a collection based on kwargs:

* Prototype: def insert_school(mongo_collection, **kwargs):
* mongo_collection will be the pymongo collection object
* Returns the new _id

---

### [32. Change school topics](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/32-update_topics.py "32. Change school topics")

Write a Python function that changes all topics of a school document based on the name:

* Prototype: def update_topics(mongo_collection, name, topics):
* mongo_collection will be the pymongo collection object
* name (string) will be the school name to update
* topics (list of strings) will be the list of topics approached in the school

---

### [33. Where can I learn Python?](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/33-schools_by_topic.py "33. Where can I learn Python?")

Write a Python function that returns the list of school having a specific topic:

* Prototype: def schools_by_topic(mongo_collection, topic):
* mongo_collection will be the pymongo collection object
* topic (string) will be topic searched

---

### [34. Log stats](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/34-log_stats.py "34. Log stats")

Write a Python script that provides some stats about Nginx logs stored in MongoDB:

* Database: logs
* Collection: nginx
* Display (same as the example):
	* first line: x logs where x is the number of documents in this collection
	* second line: Methods:
	* 5 lines with the number of documents with the method = ["GET", "POST", "PUT", "PATCH", "DELETE"]
	* one line with the number of documents with:
		* method=GET
		* path=/status
You can use this dump as data sample: [dump.zip](https://holbertonintranet.s3.amazonaws.com/uploads/misc/2020/6/645541f867bb79ae47b7a80922e9a48604a569b9.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220310%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220310T224854Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=e819a1c38601a5d88bf0fe38a72b07dc40dd17bd70119eb67110855eb368c756)

---

## Advanced Tasks

---


### [35. Optimize simple search](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/100-index_my_names.sql "35. Optimize simple search")

Write a SQL script that creates an index idx_name_first on the table names and the first letter of name.

Requirements:
* Import this table dump: [names.sql.zip](https://intranet-projects-files.s3.amazonaws.com/holbertonschool-webstack/632/names.sql.zip)
* Only the first letter of name must be indexed

---

### [36. Optimize search and score](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/101-index_name_score.sql "36. Optimize search and score")

Write a SQL script that creates an index idx_name_first_score on the table names and the first letter of name and the score.

Requirements:
* Import this table dump: [names.sql.zip](https://intranet-projects-files.s3.amazonaws.com/holbertonschool-webstack/632/names.sql.zip)
* Only the first letter of name AND score must be indexed

---

### [37. No table for a meeting](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/102-need_meeting.sql "37. No table for a meeting")

Write a SQL script that creates a view need_meeting that lists all students that have a score under 80 (strict) and no last_meeting or more than 1 month.

Requirements:
* The view need_meeting should return all students name when:
* They score are under (strict) to 80
* AND no last_meeting date OR more than a month

---

### [38. Average weighted score](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/103-average_weighted_score.sql "38. Average weighted score")

Write a SQL script that creates a stored procedure ComputeAverageWeightedScoreForUser that computes and store the average weighted score for a student.

Requirements:
* Procedure ComputeAverageScoreForUser is taking 1 input:
	* user_id, a users.id value (you can assume user_id is linked to an existing users)

---

### [39. Regex filter](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/104-find "39. Regex filter")

Write a script that lists all documents with name starting by Holberton in the collection school:

* The database name will be passed as option of mongo command

---

### [40. Top students](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/105-students.py "40. Top students")

Write a Python function that returns all students sorted by average score:

* Prototype: def top_students(mongo_collection):
* mongo_collection will be the pymongo collection object
* The top must be ordered
* The average score must be part of each item returns with key = averageScore

---

### [41. Log stats - new version](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x02-databases/106-log_stats.py "41. Log stats - new version")

Improve 34-log_stats.py by adding the top 10 of the most present IPs in the collection nginx of the database logs:

* The IPs top must be sorted

---

## Author

[Benjamin Dosch](https://github.com/BenDoschGit)
