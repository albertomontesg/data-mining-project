# Large-Scale Image Classification

In this project your task is to classify images in one of two classes according to their visual content. We will provide a labeled dataset containing two sets of images: Nature and People.

## Dataset

A set of 400 features has been extracted from each picture. We provide 80k training images (**handout_train.txt**) and 20k testing images (**handout_test.txt**), sampled rougly at the same frequency from both categories. We only provide the features for each image, from which the actual image cannot be reconstructed. Each line in the files corresponds to one image and is formatted as follows:

1. Elements are space separated.
2. The first element in the line is the class y {+1,-1} which correspond to Nature and People class, respectively.
3. The next 400 elements are real numbers which represent the feature values x_0... x_399.

## Task

The goal is to solve this classification problem using Parallel Stochastic Gradient Descent. To facilitate development, we provide you with a (small) MapReduce implementation for UNIX: The **runner.py** script allows you to run a MapReduce program and measure the performance of the produced solutions.

To create a valid MapReduce program for this task, you need to create a Python source file that contains both a mapper and a reducer function. The **mapper(key, value)** function takes as input a (key, value) tuple where key is None and value is a string. It should yield (key, value) pairs. The **reducer(key, value)** function takes as input a key and a list of values. It should yield (key, value) pairs. A skeleton of such a function is provided in the **example.py**.

You are free to implement the mapper and reducer in any way you see fit, as long as the following holds:

1. The cumulative runtime of both mappers, reducers and evaluation is limited to 15 minutes.
2. The cumulative memory limit is 1 GB.
3. Each mapper receives a key and value pair where key is always None (for consistency). Each value is a 2D NumPy array representing the subset of images passed to the mapper.
4. There will be one reducer process. All mappers should output the same key.
5. The reducer should output the weight vector that we will use to perform predictions as described below.
6. You may use the Python 2.7 standard library and **NumPy**. You are not allowed to use multithreading, multiprocessing, networking, files and sockets. In particular, **you are not allowed to use the scikit-learn library**.

## Evaluation and Grading

The prediction of your model on a test instance x will be calculated as **y' = sgn(<w, x>)**. If you decide to apply any transformation t to the given features your predictions will be given by **y = sgn(<w, t(x)>)**. If you apply transformations to the original features you have to implement a **transform** function in the given mapper template, otherwise we will not be able to transform the evaluation data using the same function and evaluate your submission. **The transform function must work with both vectors and 2D Numpy arrays**.

Based on the predictions we will calculate the predictive accuracy as the **(TP + TN)/(TP + TN + FP + FN)** where TP, TN, FP, FN are the number of true positives, true negatives, false positives and false negatives, respectively.

We will compare the F1 score of your submission to two baseline solutions: A weak one (called **baseline easy**) and a strong one (**baseline hard**). These will have the accuracy of FBE and FBH respectively, calculated as described above. Both baselines will appear in the rankings together with the score of your solutions.

Your grade on this task depends on the solution and the description that you hand in. As a rough (non-binding) guidance, if you hand in a properly-written description and your handed-in submission performs better than the easy baseline, you will obtain a grade exceeding a 4. If in addition your submission performs better than the hard baseline, you obtain a 6.

For this task, each submission will be evaluated on two datasets: a public data set for which you will see the score in the leaderboard and a private data set for which the score is kept private until hand in. Both the public and the private score of the handed in submission are used in the final grading.
