# Pandas

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
	1. [From Numpy](#0-from-numpy)
	2. [From Dictionary](#1-from-dictionary)
	3. [From File](#2-from-file)
	4. [Rename](#3-rename)
	5. [To Numpy](#4-to-numpy)
	6. [Slice](#5-slice)
	7. [Flip it and Switch it](#6-flip-it-and-switch-it)
	8. [Sort](#7-sort)
	9. [Prune](#8-prune)
	10. [Fill](#9-fill)
	11. [Indexing](#10-indexing)
	12. [Concat](#11-concat)
	13. [Hierarchy](#12-hierarchy)
	14. [Analyze](#13-analyze)
	15. [Visualize](#14-visualize)
4. [Author](#author)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* 

## Refrences

* [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html "10 minutes to pandas")
* [Complete Python Pandas Data Science Tutorial! (Reading CSV/Excel files, Sorting, Filtering, Groupby)](https://www.youtube.com/watch?v=vmEHCJofslg "Complete Python Pandas Data Science Tutorial! (Reading CSV/Excel files, Sorting, Filtering, Groupby)")

## Tasks
List of tasks with brief descriptions of each task.
Some tasks use a [Coinbase](https://drive.google.com/file/d/16MgiuBfQKzXPoWFWi2w-LKJuZ7LgivpE/view) or [Bitstamp](https://drive.google.com/file/d/15A-rLSrfZ0td7muSrYHy0WX9ZqrMweES/view) dataset.

### [0. From Numpy](https://github.com/BenDoschGit/pipeline/blob/main//0-from_numpy.py "0. From Numpy")

Write a function using the prototype from_numpy(array) that creates a pandas DataFrame from a numpy ndarray. array is the array from which you should create the DataFrame. The columns of the DataFrame should be labeled in alphabetical order and capitalized. Return the newly created DataFrame.

---

### [1. From Dictionary](https://github.com/BenDoschGit/pipeline/blob/main//1-from_dictionary.py "1. From Dictionary")

Write a script that, that creates a pandas DataFrame from a dictionary. The first column should be labeled First and have the values 0.0, 0.5, 1.0, and 1.5.
The second column should be labeled Second and have the values one, two, three, four. The rows should be labeled A, B, C, and D, respectively. The pd.DataFrame should be saved into the variable df.

---

### [2. From File](https://github.com/BenDoschGit/pipeline/blob/main//2-from_file.py "2. From File")

Write a function using the prototype from_file(filename, delimiter) that loads data from a file as a pandas DataFrame. filename is the file to load from and delimiter is the column separator. Return the loaded pandas DataFrame.

---

### [3. Rename](https://github.com/BenDoschGit/pipeline/blob/main//3-rename.py "3. Rename")

Complete the script below to perform the following:

Rename the column Timestamp to Datetime
Convert the timestamp values to datatime values
Display only the Datetime and Close columns
```
#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE

print(df.tail())
```

---

### [4. To Numpy](https://github.com/BenDoschGit/pipeline/blob/main//4-array.py "4. To Numpy")

Complete the following script to take the last 10 rows of the columns High and Close and convert them into a numpy.ndarray:

```
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

A = # YOUR CODE HERE

print(A)
```

---

### [5. Slice](https://github.com/BenDoschGit/pipeline/blob/main//5-slice.py "5. Slice")

Complete the following script to slice the pd.DataFrame along the columns High, Low, Close, and Volume_BTC, taking every 60th row:

```
#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = # YOUR CODE HERE

print(df.tail())
```

---

### [6. Flip it and Switch it](https://github.com/BenDoschGit/pipeline/blob/main//6-flip_switch.py "6. Flip it and Switch it")

Complete the following script to alter the pd.DataFrame such that the rows and columns are transposed and the data is sorted in reverse chronological order:

```
#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = # YOUR CODE HERE

print(df.tail(8))
```

---

### [7. Sort](https://github.com/BenDoschGit/pipeline/blob/main//7-high.py "7. Sort")

Complete the following script to sort the pd.DataFrame by the High price in descending order:

```
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = # YOUR CODE HERE

print(df.head())
```

---

### [8. Prune](https://github.com/BenDoschGit/pipeline/blob/main//8-prune.py "8. Prune")

Complete the following script to remove the entries in the pd.DataFrame where Close is NaN:

```
#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = # YOUR CODE HERE

print(df.head())
```

---

### [9. Fill](https://github.com/BenDoschGit/pipeline/blob/main//9-fill.py "9. Fill")

Complete the following script to fill in the missing data points in the pd.DataFrame:

The column Weighted_Price should be removed
missing values in Close should be set to the previous row value
missing values in High, Low, Open should be set to the same row’s Close value
missing values in Volume_(BTC) and Volume_(Currency) should be set to 0

```
#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE

print(df.head())
print(df.tail())
```

---

### [10. Indexing](https://github.com/BenDoschGit/pipeline/blob/main//10-index.py "10. Indexing")

Complete the following script to index the pd.DataFrame on the Timestamp column:

```
#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = # YOUR CODE HERE

print(df.tail())
```

---

### [11. Concat](https://github.com/BenDoschGit/pipeline/blob/main//11-concat.py "11. Concat")

Complete the following script to index the pd.DataFrames on the Timestamp columns and concatenate them:

Concatenate the start of the bitstamp table onto the top of the coinbase table
Include all timestamps from bitstamp up to and including timestamp 1417411920
Add keys to the data labeled bitstamp and coinbase respectively

```
#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

# YOUR CODE HERE

df = # YOUR CODE HERE

print(df)
```

---

### [12. Hierarchy](https://github.com/BenDoschGit/pipeline/blob/main//12-hierarchy.py "12. Hierarchy")

Based on 11-concat.py, rearrange the MultiIndex levels such that timestamp is the first level:

Concatenate th bitstamp and coinbase tables from timestamps 1417411980 to 1417417980, inclusive
Add keys to the data labeled bitstamp and coinbase respectively
Display the rows in chronological order

---

### [13. Analyze](https://github.com/BenDoschGit/pipeline/blob/main//13-analyze.py "13. Analyze")

Complete the following script to calculate descriptive statistics for all columns in pd.DataFrame except Timestamp:

```
#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

stats = # YOUR CODE HERE

print(stats)
```

---

### [14. Visualize](https://github.com/BenDoschGit/pipeline/blob/main//14-visualize.py "14. Visualize")

Complete the following script to visualize the pd.DataFrame:

The column Weighted_Price should be removed
Rename the column Timestamp to Date
Convert the timestamp values to date values
Index the data frame on Date
Missing values in Close should be set to the previous row value
Missing values in High, Low, Open should be set to the same row’s Close value
Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
Plot the data from 2017 and beyond at daily intervals and group the values of the same day such that:

High: max
Low: min
Open: mean
Close: mean
Volume(BTC): sum
Volume(Currency): sum

```
#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE
```

---

## Author

[Benjamin Dosch](https://github.com/BenDoschGit)
