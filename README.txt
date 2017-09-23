# naive-bayes-text-classifier

Naive Bayes Text Classifier for yelp reviews data
===============================================================================

Purdue University - Spring 2017 - CS 573 Data Mining - Homework 2
===============================================================================

Author: Parag Guruji
Email: pguruji@purdue.edu
===============================================================================
 
Python Version 2.7
===============================================================================

Directory structure:

parag_guruji/
	|---nbc.py
	|---analysis.pdf
	|---README.txt
	|---__init__.py
	|---requirements.txt
	|---output/ (OPTIONAL)
		|---question3.csv
		|---question3.png
		|---question4.csv
		|---question4.png
		|---screenshot1.png
		|---screenshot2.png

===============================================================================

usage: nbc.py [-h] [-f feature_count] [-s stopword_count] [-e evaluation]
              train_set [test_set_or_percent]

CS 573 Data Mining HW2 NBC Implementation

positional arguments:
  train_set             file-path of training set (when next positional
                        argument is a string specifying file-path of test_set
                        --------------------------OR--------------------------
                        file-path of whole yelp dataset (when next positional
                        argument is an integer specifying percent of whole
                        data to be used for training).
  test_set_or_percent   If string: filenames of testing dataset. . . . . . . .
                        If integer: percent of data to be used for training
                        (default: 50)

optional arguments:
  -h, --help            show this help message and exit
  -f feature_count, --feature_count feature_count
                        number of features (default: 500)
  -s stopword_count, --stopword_count stopword_count
                        number of most frequent words to be considered as
                        stopwords (default: 100)
  -e evaluation, --evaluation evaluation
                        file-path of whole data set to be used for performance
                        evaluation as specified in Q3 & Q4. The output files
                        are stored in a subdirectory of current working
                        directory named output (created if doesn't exist)
                        (default: None)

===============================================================================
Examples: 
nbc.py train-set.dat test-set.dat
nbc.py train-set.dat test-set.dat -e whole-data.dat
nbc.py train-set.dat 50 -f 1000 -s 200

===============================================================================
