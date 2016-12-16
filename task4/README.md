# Large-Scale Bandit Optimization

## Task Description

The goal of this task is to learn a policy that explores and exploits among available choices in order to learn user preferences and recommend news articles to users. For this task we will use real-world log data shared by Yahoo!. The data was collected over a 10 day period and consists of **45 million log** lines that capture the interaction between user visits and 271 news articles, one of which was randomly displayed for every user visit. In each round, you are given a user context and a list of available articles for recommendation.

Your task is to then select one article from this pool such to maximize the **click-through rate (clicks/impressions)**. If the article you selected matches the one displayed to the user (in the log file), then your policy is evaluated for that log line. Otherwise, the line is discarded. Since the articles were displayed uniformly at random during the data collection phase, there is approximately 1 in 20 chance that any given line will be evaluated.

## Dataset Description

In the handout for this project, you will find the 100000 lines of log data **webscope-logs.txt**, where each line is formated as follows:

1. timestamp: integer
2. user features: 6 dimensional vector of doubles
3. available articles: list of article IDs that are available for selection

In addition, you are given access to **webscope-articles.txt**, where each line is formated as follows:

1. ArticleID feature1 feature2 feature3 feature4 feature5 feature6

To evaluate your policy you can use (**policy.py** and **runner.py**). Your task is to complete the functions **recommend**, **update** and **set_articles** in the policy file. We will first call the set_articles method and pass in all the article features. Then for every line in in **webscope-logs.txt** the provided runner will call your **recommend** function. If your chosen article matches the displayed article in the log, the result of choosing this article (click or no click) is fed back to your policy and you have a chance to update the current model accordingly. This is achieved via the **update** method.

You are free to implement the policy in any way you see fit, as long as the following holds:

1. Your policy always returns an article from the provided list. Failing to do so will result in an exception and the execution will then be aborted.
2. The cummulative runtime is 30 minutes.
3. The cummulative memory limit is 4 GB.

## Evaluation and Grading

An iteration is evaluated only if you have chosen the same article as the one chosen by the random policy used to generate the data set. Only those selections of your policy that match the log line are counted as an impression. Based on the number of impressions and clicks we calculate the click-through rate.

We will compare the score of your submission to two baseline solutions: A weak one (called **baseline easy**) and a strong one (**baseline hard**). These will have the **quantization error** of FBE and FBH respectively, calculated as described above. Both baselines will appear in the rankings together with the score of your solutions.

Your grade on this task depends on the solution and the description that you hand in. As a rough (non-binding) guidance, if you hand in a properly-written description and your handed-in submission performs better than the easy baseline, you will obtain a grade exceeding a 4. If in addition your submission performs better than the hard baseline, you obtain a 6.
