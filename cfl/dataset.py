

class Dataset():

    def __init__(self, X, Y, dataset_label, experiment_saver=None):
        self.X = X
        self.Y = Y
        self.dataset_label = dataset_label
        self.to_save = experiment_saver is not None
        self.saver = None
        if self.to_save:
            self.saver = experiment_saver.get_new_dataset_saver(dataset_label)
        self.pyx = None

    # TODO: add other attributes/methods that would be helpful to keep together with a dataset