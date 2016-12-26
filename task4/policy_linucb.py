import logging

import numpy as np
from numpy import linalg

logger = logging.getLogger(__name__)

np.random.seed(23)

method = 'LinUCB'
DEBUG = 1


class LinUCB:
    """ LinUCB class """
    def __init__(self):
        # upper bound coefficient
        self.alpha = 2.5 # if worse -> 2.9, 2.8 1 + np.sqrt(np.log(2/delta)/2)
        r1 = .5 # if worse -> 0.7, 0.8
        r0 = -15 # if worse, -19, -21
        self.r = (r0, r1)
        # dimension of user features = d
        self.d = 6
        # Ma : collection of matrix to compute disjoint part for each article a, d*d
        self.Ma = dict()
        # MaI : store the inverse of all Ma matrix
        self.MaI = dict()
        # ba : collection of vectors to compute disjoin part, d*1
        self.ba = dict()

        self.a_max = 0

        self.wa = dict()
        self.z = None
        self.zT = None

    def set_articles(self, art):
        """ Initialize all the variables for the articles keys given """
        for key in art:
            self.Ma[key] = np.identity(self.d)
            self.ba[key] = np.zeros((self.d, 1))
            self.MaI[key] = np.identity(self.d)
            self.wa[key] = np.zeros((self.d, 1))

    def update(self, reward):
        """ Update the parameters given the result """
        if reward == -1:
            return
        r = self.r[reward]
        self.Ma[self.a_max] += self.z.dot(self.zT)
        self.ba[self.a_max] += r * self.z
        self.MaI[self.a_max] = linalg.solve(self.Ma[self.a_max], np.identity(self.d))
        self.wa[self.a_max] = self.MaI[self.a_max].dot(self.ba[self.a_max])


    def recommend(self, timestamp, user_features, articles):
        """ Recommend the next article given the possible choices """
        zaT = np.array([user_features])
        za = zaT.T

        UCBa = np.array([float(np.dot(zaT, self.wa[article]) + self.alpha * np.sqrt(np.dot(zaT.dot(self.MaI[article]), za))) for article in articles])
        self.a_max = articles[divmod(UCBa.argmax(), UCBa.shape[0])[1]]
        self.z = za
        self.zT = zaT

        return self.a_max


algorithms = {
    'LinUCB': LinUCB()
}

algorithm = algorithms[method]
set_articles = algorithm.set_articles
update = algorithm.update
recommend = algorithm.recommend
