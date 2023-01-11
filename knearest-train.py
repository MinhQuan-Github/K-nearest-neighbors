import preprocessing


class KNearestTrain:
    def __init__(self):
        self.prep = preprocessing.Preprocessing()
        self.data = self.prep.get_data_training()

