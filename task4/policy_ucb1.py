import logging

import numpy as np
from numpy.linalg import inv

logger = logging.getLogger(__name__)

np.random.seed(23)

method = 'UCB1'
DEBUG = 1

class UCB1(object):
    # Define variables
    index_map = dict()
    index_article = dict()
    mu = None
    n = None
    t = 0

    r_article_idx = None
    r_article = None
    K = None

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
        self.K = num_articles

    @classmethod
    def recommend(self, time, user_features, choices):
        """ Recommend the next article given the possible choices """

        idx = [self.index_map[c] for c in choices]
        n = self.n[idx]
        mu = self.mu[idx]

        if np.sum(n==0) > 0:
            r_article = choices[np.argmin(n)]
        else:
            self.t += 1
            UCB = mu + np.sqrt(2 * np.log(self.t) / n)
            r_article = choices[np.argmax(UCB)]


        self.r_article = r_article
        self.r_article_idx = self.index_map[r_article]
        return r_article

    @classmethod
    def update(self, reward):
        """ Update the parameters given the reward """
        reward += 1
        self.n[self.r_article_idx] += 1
        self.mu[self.r_article_idx] += 1/self.n[self.r_article_idx]* (reward-self.mu[self.r_article_idx])



algorithms = {
    'UCB1': UCB1
}

algorithm = algorithms[method]
set_articles = algorithm.set_articles
update = algorithm.update
recommend = algorithm.recommend
