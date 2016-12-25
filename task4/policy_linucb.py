import logging

import numpy as np
from numpy import linalg

logger = logging.getLogger(__name__)

np.random.seed(23)

method = 'LinUCB'
DELTA = .5
ALPHA = 1 + np.sqrt(np.log(2 / DELTA) / 2)
DEBUG = 1

logger.info('alpha: {}'.format(ALPHA))



# lin UCB
class LinUCB:
    def __init__(self):
        # upper bound coefficient
        self.alpha = 3 # if worse -> 2.9, 2.8 1 + np.sqrt(np.log(2/delta)/2)
        r1 = 0.5 # if worse -> 0.7, 0.8
        r0 = -20 # if worse, -19, -21
        self.r = (r0, r1)
        # dimension of user features = d
        self.d = 6
        # Aa : collection of matrix to compute disjoint part for each article a, d*d
        self.Aa = dict()
        # AaI : store the inverse of all Aa matrix
        self.AaI = dict()
        # ba : collection of vectors to compute disjoin part, d*1
        self.ba = dict()

        self.a_max = 0

        self.theta = dict()
        self.x = None
        self.xT = None

    def set_articles(self, art):
        # init collection of matrix/vector Aa, Ba, ba
        for key in art:
            self.Aa[key] = np.identity(self.d)
            self.ba[key] = np.zeros((self.d, 1))
            self.AaI[key] = np.identity(self.d)
            self.theta[key] = np.zeros((self.d, 1))

    def update(self, reward):
        if reward == -1:
            return
        r = self.r[reward]
        self.Aa[self.a_max] += self.x.dot(self.xT)
        self.ba[self.a_max] += r * self.x
        self.AaI[self.a_max] = linalg.solve(self.Aa[self.a_max], np.identity(self.d))
        self.theta[self.a_max] = self.AaI[self.a_max].dot(self.ba[self.a_max])


    def recommend(self, timestamp, user_features, articles):
        xaT = np.array([user_features])
        xa = xaT.T

        pa = np.array([float(np.dot(xaT, self.theta[article]) + self.alpha * np.sqrt(np.dot(xaT.dot(self.AaI[article]), xa))) for article in articles])
        self.a_max = articles[divmod(pa.argmax(), pa.shape[0])[1]]
        self.x = xa
        self.xT = xaT

        return self.a_max


# class LinUCB(object):
#     # Define variables
#     M = dict()
#     M_inv = dict()
#     b = dict()
#     x_t = None      # Last recommended article
#     z_t = None      # User features from last recommended article
#
#     @classmethod
#     def set_articles(self, articles):
#         """ Initialize all the variables for the articles features given """
#         num_articles = len(articles)
#         num_features = len(articles[articles.keys()[0]])
#         if DEBUG > 0:
#             logger.info('Number of articles: {}'.format(num_articles))
#             logger.info('Number of features: {}'.format(num_features))
#
#         articles_ids = articles.keys()
#         for art in articles_ids:
#             self.M[art] = None
#             self.M_inv[art] = None
#             self.b[art] = None
#
#     @classmethod
#     def recommend(self, time, user_features, choices):
#         """ Recommend the next article given the possible choices """
#         z_t = np.array(user_features)
#         dim = z_t.shape[0]
#         max_UCB = 0
#         max_article = None
#         for x in choices:
#             if self.M[x] is None:
#                 # In the case `x` is new
#                 self.M[x] = np.eye(dim)
#                 self.M_inv[x] = np.eye(dim)
#                 self.b[x] = np.zeros((dim,))
#
#             M_inv_x, b_x = self.M_inv[x], self.b[x]
#             w_x = M_inv_x.dot(b_x)
#             UCB_x = w_x.dot(z_t) + ALPHA * np.sqrt(z_t.T.dot(M_inv_x).dot(z_t))
#             if max_article is None or UCB_x > max_UCB:
#                 max_UCB = UCB_x
#                 max_article = x
#
#         self.x_t = max_article
#         self.z_t = z_t
#         return max_article
#
#
#     @classmethod
#     def update(self, reward):
#         """ Update the parameters given the reward """
#         if reward == -1:
#             return
#         z_t = self.z_t.reshape(-1, 1)
#         self.M[self.x_t] += z_t.dot(z_t.T)
#         self.M_inv[self.x_t] = inv(self.M[self.x_t])
#         self.b[self.x_t] += self.z_t * reward


algorithms = {
    'LinUCB': LinUCB()
}

algorithm = algorithms[method]
set_articles = algorithm.set_articles
update = algorithm.update
recommend = algorithm.recommend
