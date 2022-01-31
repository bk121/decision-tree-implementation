from scipy.stats import mode


class Leaf:
    def __init__(self, labels):
        self.value = mode(labels)[0]

    def predict(self, data):
        return self.value[0]
