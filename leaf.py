from scipy.stats import mode


class Leaf:
    def __init__(self, labels):
        self.value = mode(labels)

    def predict(self, data):
        return self.value
