import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Preprocessing:
    def __init__(self):
        self.data_path = 'datasets/the_trang.csv'
        self.label_path = 'datasets/the_trang_labels.csv'
        self.data = pd.read_csv(self.data_path).values
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = pd.read_csv(self.label_path).values

    def drawData(self, isShow=False, show_label=False):
        plt.title("K-nearest-neighbors")
        plt.xlabel("Heights")
        plt.ylabel("Weights")
        if show_label is None:
            plt.scatter(self.data[:, 0], self.data[:, 1])
        else:
            plt.scatter(self.data[:, 0], self.data[:, 1], c=self.data[:, 2])
        if isShow:
            plt.show()

    def drawPoints(self, points, isShow=False, cs='rx'):
        plt.plot(points[:, 0], points[:, 1], label=cs)
        if isShow:
            plt.show()

    def normalization(self):
        scaler = [1, 1]
        scaler_save = []

        for i in range(self.data.shape[1] - 1):
            min_cols = np.min(self.data[:, i])
            max_cols = np.max(self.data[:, i])
            self.data[:, i] = scaler[i] * (self.data[:, i] - min_cols) / (max_cols - min_cols)
            scaler_save.append([min_cols, max_cols])
        scaler_save = np.array(scaler_save)
        np.save('models/scaler_min_max.npy', scaler_save)
        np.save('models/scaler_label.npy', self.labels)
        np.save('models/scaler_data.npy', self.data)
        print("normalize succeeded")

    def get_data_training(self):
        self.normalization()
        return self.data, self.labels


if __name__ == '__main__':
    prep_objc = Preprocessing()
    prep_objc.normalization()
    print(prep_objc.labels)
    prep_objc.drawData(isShow=True, show_label=True)
