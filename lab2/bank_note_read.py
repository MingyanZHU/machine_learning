import numpy as np
import pandas as pd


class BankNoteRead(object):
    def __init__(self):
        self.data_set = pd.read_csv("./data_banknote_authentication.csv")
        self.x = self.data_set.drop('a', axis=1)
        self.y = self.data_set['a']
    def get_data(self):
        return np.array(self.x, dtype=float), np.array(self.y)

def main():
    bn = BankNoteRead()
    print(bn.data_set.head())
    print(bn.data_set.isnull().sum())
    print(bn.x.head())
    print(bn.y.head())

if __name__ == '__main__':
    main()
