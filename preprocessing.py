import os
from matplotlib.image import imread
import numpy as np


class PreProcessing:

    images_train = np.array([])
    images_test = np.array([])
    labels_train = np.array([])
    labels_test = np.array([])
    unique_train_label = np.array([])
    map_train_label_indices = dict()

    def __init__(self,data_src):
        self.data_src = data_src
        print("Loading Geological Similarity Dataset...")
        self.images_train, self.images_test, self.labels_train, self.labels_test = self.preprocessing(0.9)
        self.unique_train_label = np.unique(self.labels_train)
        self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label) for label in
                                        self.unique_train_label}
        print('Preprocessing Done. Summary:')
        print("Images train :", self.images_train.shape)
        print("Labels train :", self.labels_train.shape)
        print("Images test  :", self.images_test.shape)
        print("Labels test  :", self.labels_test.shape)
        print("Unique label :", self.unique_train_label)

    def normalize(self,x):
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x - min_val) / (max_val - min_val)
        return x

    def read_dataset(self):
        X = []
        y = []
        for directory in os.listdir(self.data_src):
            try:
                for pic in os.listdir(os.path.join(self.data_src, directory)):
                    img = imread(os.path.join(self.data_src, directory, pic))
                    X.append(np.squeeze(np.asarray(img)))
                    y.append(directory)
            except Exception as e:
                print('Failed to read images from Directory: ', directory)
                print('Exception Message: ', e)
        print('Dataset loaded successfully.')
        return X,y

    def preprocessing(self,train_test_ratio):
        X, y = self.read_dataset()
        labels = list(set(y))
        label_dict = dict(zip(labels, range(len(labels))))
        Y = np.asarray([label_dict[label] for label in y])
        X = [self.normalize(x) for x in X]                                  # normalize images

        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = []
        y_shuffled = []
        for index in shuffle_indices:
            x_shuffled.append(X[index])
            y_shuffled.append(Y[index])

        size_of_dataset = len(x_shuffled)
        n_train = int(np.ceil(size_of_dataset * train_test_ratio))
        return np.asarray(x_shuffled[0:n_train]), np.asarray(x_shuffled[n_train + 1:size_of_dataset]), np.asarray(
            y_shuffled[0:n_train]), np.asarray(y_shuffled[
                                               n_train + 1:size_of_dataset])


    def get_triplets(self):
        label_l, label_r = np.random.choice(self.unique_train_label, 2, replace=False)
        a, p = np.random.choice(self.map_train_label_indices[label_l],2, replace=False)
        n = np.random.choice(self.map_train_label_indices[label_r])
        return a, p, n

    def get_triplets_batch(self,n):
        idxs_a, idxs_p, idxs_n = [], [], []
        for _ in range(n):
            a, p, n = self.get_triplets()
            idxs_a.append(a)
            idxs_p.append(p)
            idxs_n.append(n)
        return self.images_train[idxs_a,:], self.images_train[idxs_p, :], self.images_train[idxs_n, :]

