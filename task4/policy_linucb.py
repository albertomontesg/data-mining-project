import logging

import numpy as np
from numpy.linalg import inv

logger = logging.getLogger(__name__)

np.random.seed(23)

method = 'LinUCB'
DELTA = .9
# ALPHA = 1 + np.sqrt(np.log(2 / DELTA) / 2)
ALPHA = .2
DEBUG = 1

logger.info('alpha: {}'.format(ALPHA))

class LinUCB(object):
    # Define variables
    X = dict()      # Article features
    M = dict()
    M_inv = dict()
    b = dict()
    x_t = None      # Last recommended article
    z_t = None      # User features from last recommended article

    @classmethod
    def set_articles(self, articles):
        """ Initialize all the variables for the articles features given """
        num_articles = len(articles)
        num_features = len(articles[articles.keys()[0]])
        if DEBUG > 0:
            logger.info('Number of articles: {}'.format(num_articles))
            logger.info('Number of features: {}'.format(num_features))

        articles_ids = articles.keys()
        for art in articles_ids:
            self.X[art] = articles[art]
            self.M[art] = np.eye(num_features)
            self.M_inv[art] = np.eye(num_features)
            self.b[art] = np.zeros((num_features,))

    @classmethod
    def recommend(self, time, user_features, choices):
        """ Recommend the next article given the possible choices """
        z_t = np.array(user_features)
        max_UCB = 0
        max_article = None
        for art in choices:
            M_inv_x, b_x = self.M_inv[art], self.b[art]
            w_x = M_inv_x.dot(b_x)
            UCB_x = w_x.dot(z_t) + ALPHA * np.sqrt(z_t.T.dot(M_inv_x).dot(z_t))
            if max_article is None or UCB_x > max_UCB:
                max_UCB = UCB_x
                max_article = art

        self.x_t = max_article
        self.z_t = z_t
        return max_article


    @classmethod
    def update(self, reward):
        """ Update the parameters given the reward """
        if reward == -1:
            return
        z_t = self.z_t.reshape(-1, 1)
        self.M[self.x_t] += z_t.dot(z_t.T)
        self.M_inv[self.x_t] = inv(self.M[self.x_t])
        self.b[self.x_t] = self.z_t * reward


algorithms = {
    'LinUCB': LinUCB
}

algorithm = algorithms[method]
set_articles = algorithm.set_articles
update = algorithm.update
recommend = algorithm.recommend
