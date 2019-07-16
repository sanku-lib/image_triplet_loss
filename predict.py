import tensorflow as tf
import numpy as np
import gc
from preprocessing import PreProcessing
from model import SiameseNetwork


class Predict:

    def __init__(self):
        data_src = './data_repository/geological_similarity/'
        self.model_path = './trained_model/model_triplet/'

        self.dataset = PreProcessing(data_src)
        self.model = SiameseNetwork()

        # Define Tensor
        self.img_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 3], name='img')
        self.net = self.model.conv_net(self.img_placeholder, reuse=False)
        self.normalized_training_vectors = self.generate_db_normed_vectors()
        print('Prediction object loaded successfully.')

    # Compute Vector representation for each training images and normalize those
    def generate_db_normed_vectors(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            saver.restore(sess, self.model_path + "model.ckpt")
            train_vectors = sess.run(self.net, feed_dict={self.img_placeholder: self.dataset.images_train})
        del self.dataset.images_train
        gc.collect()
        normalized_train_vectors = train_vectors / np.linalg.norm(train_vectors, axis=1).reshape(-1, 1)
        return normalized_train_vectors

    # Find k nearest neighbour using cosine similarity
    def find_k_nn(self,normalized_train_vectors, vec, k):
        dist_arr = np.matmul(normalized_train_vectors, vec.T)
        return np.argsort(-dist_arr.flatten())[:k]

    def predict(self, im, k = 10):
        # run the test image through the network to get the test features
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            saver.restore(sess, self.model_path + "model.ckpt")
            search_vector = sess.run(self.net, feed_dict={self.img_placeholder: [im]})
        normalized_search_vec = search_vector / np.linalg.norm(search_vector)
        candidate_index = self.find_k_nn(self.normalized_training_vectors, normalized_search_vec, k)
        return list(candidate_index)


if __name__ == "__main__":
    k = 10
    predict = Predict()
    idx = np.random.randint(0, len(predict.dataset.images_test))
    test_image = predict.dataset.images_test[idx]
    print("Index of Test Image:  ", idx)
    index_similar_images = predict.predict(test_image, k)
    print('Index of Similar Images: ',index_similar_images)




