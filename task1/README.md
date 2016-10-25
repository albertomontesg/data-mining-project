# Approximate Nearest Neighbour with Locality Sensitive Hashing

In this project, we are interested in detecting near-duplicate videos. Suppose you work for a company which offers a web service that allows users to upload various cat videos. If users upload a copyrighted video, the company can be sued by the copyright owner. Your task is to develop an efficient method to to detect whether a video is a near-duplicate of a copyrighted video so that lawsuits can be avoided.

## Input and Output Specification

We will use Jaccard similarity based on the video features as the similarity metric. You are given two text files:

* **handout_shingles.txt**: Each line contains the features for one video file and is formatted as follows: **VIDEO_XXXXXXXXX** followed by a list of space delimited integers in range **[0, 8192]**. You can consider them equivalent to shingles in the context of near-duplicate document retrieval.
* **handout_duplicates.txt**: Each line contains a pair of near duplicates (videos that are at least 85% similar according to the Jaccard similarity). Each line is a tab-delimited pair of integers **where the first integer is always smaller**. This file is used to measure the error of your output as described below.

Your goal is to develop a Locality Sensitive Hashing program in the MapReduce setting. To facilitate development, we provide you with a (small) MapReduce implementation: The **runner.py** script allows you to run a MapReduce program and measure the performance of the produced solutions.

To create a valid MapReduce program for this task, you need to create a Python source file that contains both a mapper and a reducer function. The **mapper(key, value)** function takes as input a (key, value) tuple where key is None and value is a string. It should yield (key, value) pairs. The **reducer(key, value)** function takes as input a key and a list of values. It should yield (key, value) pairs. A skeleton of such a function is provided in the **example.py**.

You are free to implement the mapper and reducer in any way you see fit, as long as the following holds:

1. The maximum number of hash functions per mapper used is 1024.
2. The cummulative runtime of both mappers and reducers is limited to 10 minutes.
3. Each mapper receives a key and value pair where key is always None (for consistency). Each value is one line of input as described above.
4. Reducer should output a key, value pair of two integers representing the ID's of duplicated videos, the smaller ID being the key and the larger the value.
5. You may use the Python 2.7 standard library and **NumPy**. You are not allowed to use multithreading, multiprocessing, networking, files and sockets.

## Evaluation and Grading

For each line of the output of your algorithm we check whether the reported videos are in fact at least 85% similar. If they are, this output will count as one true positive. If they are not, it will count as one false positive. In addition, each pair of approximate neighbors that was not reported by your algorithm will count as one false negative. Given the number of true positives, false positives and false negatives, TP, FP and FN respectively, we can calculate:

1. **Precision**, also referred to as Positive predictive value (PPV), as **P = TP/(TP+FP)**.
2. **Recall**, also referred to as the True Positive Rate or Sensitivity, as **R = TP/(TP+FN)**.

Given precision and recall we will calculate the F1 score defined as **F1 = 2PR/(P + R)**. We will compare the F1 score of your submission to two baseline solutions: A weak one (called **baseline easy**) and a strong one (**baseline hard**). These will have the F1 score of FBE and FBH respectively, calculated as described above. Both baselines will appear in the rankings together with the F1 score of your solutions.

Your grade on this task depends on the solution and the description that you hand in. As a rough (non-binding) guidance, if you hand in a properly-written description and your handed-in submission performs better than the easy baseline, you will obtain a grade exceeding a 4. If in addition, your submission beats the hard baseline, you obtain a 6.


## Description of the solution

The current solution tries to solve the Approximate Nearest Neighbour using the Locality Sensitive Hashing with a Map Reduce Implementation.

In the given solution, the map function is responsible to compute a Signature hashing matrix and then hash b number of r rows into different buckets. The buckets id and the value of the video shingles passed to the mapper are returned by this one, to be grouped all the videos that have end up with the same LSH.
The reducer function receive for each bucket_id a list of videos which consist on the ones which are predicted to be closest. To ensure the distance and avoid false positives (the desired similarity is 85%), the Jacard distance is computed between all the combinations of the input list, and returns only those who has a similarity higher than the required.

The values of r and b used in the LSH determine the success of the implementation. To avoid false negative, I have chosen values of r and b that setup the threshold lower than the requested (in this case r=15 b=10 threshold=76%) and so the probability to have false negatives at 85% similarity is considerably reduced.
Phit(.85) = 1-(1-.85^10)^15 = .962
With this values and the exact computation of the Jacard similarity at the reducer it can be reduced drastically the FN and FP achieving finally a 1.0 score without any considerable tradeoff (only with 150 hash functions and no increase of the execution time due to a lower threshold).
