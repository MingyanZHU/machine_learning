import numpy as np
import pandas as pd


class BloodProcessing(object):
    def __init__(self):
        self.data_set = pd.read_csv("./blood.csv") 
        self.x = self.data_set.drop('whether he/she donated blood in March 2007', axis=1)
        self.y = self.data_set['whether he/she donated blood in March 2007']
    def get_data(self):
        return np.array(self.x, dtype=float), np.array(self.y)
def main():
    bd = BloodProcessing()
    print(bd.x.head())
    print(bd.y.head())
    x, y = bd.get_data()
    print(x.shape)
    print(y.shape)
    print(type(x[1][1]))
    # print(np.array(bd.x))
    # print(np.array(bd.y))

if __name__ == '__main__':
    main()