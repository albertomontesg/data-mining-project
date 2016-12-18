import logging

import numpy as np
from numpy.linalg import inv

logger = logging.getLogger(__name__)

np.random.seed(23)

method = 'UCB1'
delta = .1
alpha = 1 + np.sqrt(np.log(2/delta)/2)
DEBUG = 1

class UCB1(object):
    # Define variables
    index_map = dict()
    index_article = dict()
    mu = None
    n = None
    k = 0
    num_articles = None
    recommended_article = None

    @classmethod
    def set_articles(self, articles):
        """ Initialize all the variables for the articles features given """
        num_articles = len(articles)
        num_features = len(articles[articles.keys()[0]])
        if DEBUG > 0:
            logger.info('Number of articles: {}'.format(num_articles))
            logger.info('Number of features: {}'.format(num_features))

        articles_keys = articles.keys()
        self.index_map = {articles_keys[i]: i for i in range(num_articles)}
        self.index_article = {i: articles_keys[i] for i in range(num_articles)}
        self.mu = np.zeros((num_articles,))
        self.n = np.zeros((num_articles,))
        self.num_articles = num_articles



    @classmethod
    def recommend(self, time, user_features, choices):
        """ Recommend the next article given the possible choices """
        if self.k < self.num_articles:
            recommendation = self.index_map.keys()[self.k]
            self.recommended_article = self.k
            self.k += 1
            return recommendation

        UCB_i = self.mu + np.sqrt(2 * np.log(time) / self.n)
        indx = np.argmax(UCB_i)
        return self.index_article[indx]


    @classmethod
    def update(self, reward):
        """ Update the parameters given the reward """
        self.n[self.recommended_article] += 1
        self.mu[self.recommended_article] += 1/self.n[self.recommended_article]* (reward-self.mu[self.recommended_article])



algorithms = {
    'UCB1': UCB1
}

algorithm = algorithms[method]
set_articles = algorithm.set_articles
update = algorithm.update
recommend = algorithm.recommend
