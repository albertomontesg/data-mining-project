import logging

import numpy as np
from numpy.linalg import inv

logger = logging.getLogger(__name__)

np.random.seed(23)

method = 'LinUCB'
delta = .1
alpha = 1 + np.sqrt(np.log(2/delta)/2)
DEBUG = 1


class LinUCB(object):
    # Define variables
    index_map = dict()
    Z = None
    M = None
    b = None
    x_t = None
    z_t = None

    @classmethod
    def set_articles(self, articles):
        """ Initialize all the variables for the articles features given """
        num_articles = len(articles)
        num_features = len(articles[articles.keys()[0]])
        if DEBUG > 0:
            logger.info('Number of articles: {}'.format(num_articles))
            logger.info('Number of features: {}'.format(num_features))

        articles_keys = articles.keys()
        Z = []
        M = np.zeros((num_articles, num_features, num_features))
        b = np.zeros((num_articles, num_features))

        for i in range(num_articles):
            key = articles_keys[i]
            self.index_map[key] = i
            Z.append(articles[key])
            M[i] = np.eye(num_features)

        self.Z = np.array(Z)
        self.M = M
        self.b = b


    @classmethod
    def recommend(self, time, user_features, choices):
        """ Recommend the next article given the possible choices """
        idx = [self.index_map[k] for k in choices]
        n_choices = len(idx)

        M_x = self.M[idx,:,:]
        b_x = self.b[idx,:]

        w_x = np.zeros(b_x.shape)
        UCB_x = np.zeros((n_choices,))
        z_t = np.array(user_features)

        for i in range(n_choices):
            w_x[i] = inv(M_x[i]).dot(b_x[i])
            UCB_x[i] = w_x[i].dot(z_t) + alpha * np.sqrt(z_t.T.dot(inv(M_x[i])).dot(z_t))

        if DEBUG > 1:
            logger.info(UCB_x)
        recommendation = np.argmax(UCB_x)
        self.x_t = choices[recommendation]
        self.z_t = z_t
        return choices[recommendation]


    @classmethod
    def update(self, reward):
        """ Update the parameters given the reward """
        # reward = reward + 1
        if DEBUG > 1:
            logger.info(reward)
        # Get the index of the previously recommended article
        idx = self.index_map[self.x_t]
        z_t = self.z_t.reshape(-1, 1)
        self.M[idx] += z_t.dot(z_t.T)
        self.b[idx] += self.z_t * reward


algorithms = {
    'LinUCB': LinUCB
}

algorithm = algorithms[method]
set_articles = algorithm.set_articles
update = algorithm.update
recommend = algorithm.recommend
