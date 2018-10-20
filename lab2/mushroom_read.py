import numpy as np
import pandas as pd
from sklearn import preprocessing
class MushroomProcessing(object):
    def __init__(self):
        self.dataset = pd.read_csv("./mushrooms.csv")
        self.x = self.dataset.drop('class', axis=1)
        self.y = self.dataset['class']
        Enconder_x = preprocessing.LabelEncoder()
        for col in self.x.columns:
            self.x[col] = Enconder_x.fit_transform(self.x[col])
        Enconder_y = preprocessing.LabelEncoder()
        self.y = Enconder_y.fit_transform(self.y)
        self.x = pd.get_dummies(self.x,columns=self.x.columns,drop_first=True)
        self.x = np.array(self.x)
        self.x_scaled = preprocessing.scale(self.x)
    def get_data(self):
        return self.x_scaled, self.y
