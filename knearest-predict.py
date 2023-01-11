from scipy.spatial.distance import cdist
import numpy as np


class KNearestPredict:
    def __init__(self):
        self.min_max_scaler = np.load('models/scaler_min_max.npy')
        self.label = np.load('models/scaler_label.npy', allow_pickle=True)
        self.data = np.load('models/scaler_data.npy')
        self.data = np.array(self.data, dtype=np.float32)
        self.k = self.getNumberOfNearestPredictors()

    def input(self):
        height_input = int(input("Enter height of person: "))
        weight_input = int(input("Enter weight of person: "))
        return height_input, weight_input

    def getNumberOfNearestPredictors(self):
        return 10

    def find_k_nearest(self, distances):
        mask = np.argsort(distances, axis=1) < self.k
        return self.data[mask[0]]

    def data_labeling(self, k_nearests):
        classes, counts = np.unique(k_nearests[:, 2], return_counts=True)
        label = classes[np.argmax(counts, axis=0)]
        return label

    def predict(self, height_input=175, weight_input=72):
        height_norm = (height_input - self.min_max_scaler[0][0]) / (self.min_max_scaler[0][1] - self.min_max_scaler[0][0])
        weight_norm = (weight_input - self.min_max_scaler[1][0]) / (self.min_max_scaler[1][1] - self.min_max_scaler[1][0])

        # calculate distance
        distances = cdist(np.array([[height_norm, weight_norm]]), self.data[:, [0, 1]])

        # find k-nearest distances
        k_nearests = self.find_k_nearest(distances)

        # labeling data
        label = self.data_labeling(k_nearests)

        # label output
        print("Label: ", self.label[int(label), :])


if __name__ == '__main__':
    pred_objc = KNearestPredict()
    height, weight = pred_objc.input()
    pred_objc.predict(height, weight)
