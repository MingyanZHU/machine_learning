import numpy as np

class GaussianMixtureModel(object):
    """ 高斯混合聚类EM算法 """
    def __init__(self, k=3, delta=1e-6):
        self.k = k
        self.delta = delta
        # TODO