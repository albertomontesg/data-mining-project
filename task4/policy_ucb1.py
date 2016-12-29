import logging

import numpy as np
from numpy.linalg import inv

logger = logging.getLogger(__name__)

np.random.seed(23)

method = 'UCB1'
CHOICES = 20
DEBUG = 1

class UCB1(object):

    def __init__(self):
        # Define variables
        self.mu = None
        self.n = None
        self.t = 0

        r0 = -15
        r1 = .5
        self.r = (r0, r1)

        self.r_index = None

    def set_articles(self, articles):
        """ Initialize all the variables for the articles features given """
        num_articles = len(articles)
        num_features = len(articles[articles.keys()[0]])
        if DEBUG > 0:
            logger.info('Number of articles: {}'.format(num_articles))
            logger.info('Number of features: {}'.format(num_features))

        self.mu = np.zeros((CHOICES,))
        self.n = np.zeros((CHOICES,))

    def recommend(self, time, user_features, choices):
        """ Recommend the next article given the possible choices """
        n = self.n
        mu = self.mu
        self.t += 1

        # Try all choices once
        if self.t <= CHOICES:
            self.r_index = self.t - 1
        else:
            UCB = mu + np.sqrt(2 * np.log(self.t) / n)
            self.r_index = np.argmax(UCB) % len(choices)

        return choices[self.r_index]

    def update(self, reward):
        """ Update the parameters given the reward """
        if reward == -1:
            # If the line does not match with the policy do not take into account
            self.t -= 1
            return

        r = self.r[reward]
        j = self.r_index
        self.n[j] += 1
        self.mu[j] += 1 / self.n[j] * (r - self.mu[j])



algorithms = {
    'UCB1': UCB1()
}

algorithm = algorithms[method]
set_articles = algorithm.set_articles
update = algorithm.update
recommend = algorithm.recommend
