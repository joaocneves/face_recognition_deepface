import itertools
import math
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn import metrics

class LFW:
    path = 'lfw'
    pairs = 'lfw/pairs.txt'
    matchedPairs = []
    numMatchedPairs = 0
    mismatchedPairs = []
    numMisMatchedPairs = 0

    def __init__(self, pairs_file, img_path):

        self.pairs_file = pairs_file
        self.img_path = img_path
        self.create_pairs_array(self.img_path, self.pairs_file)
        self.dict_img_embeddings = {}

    def create_pairs_array(self, img_path, pairsList):

        file = open(pairsList, 'r')
        lines = file.readlines()

        for line in lines:

            str = line.split('\t')

            if len(str) == 3:
                img1 = os.path.join(img_path, str[0], str[0] + '_' + str[1].zfill(4) + '.png')
                img2 = os.path.join(img_path, str[0], str[0] + '_' + str[2].rstrip().zfill(4) + '.png')
                self.matchedPairs.append([img1, img2])
                self.numMatchedPairs += 1

            elif len(str) == 4:
                img1 = os.path.join(img_path, str[0], str[0] + '_' + str[1].zfill(4) + '.png')
                img2 = os.path.join(img_path, str[2], str[2] + '_' + str[3].rstrip().zfill(4) + '.png')
                self.mismatchedPairs.append([img1, img2])
                self.numMisMatchedPairs += 1

    def _load_imgs(self, img_list, preprocess_input_function):

        """

        Load a set of images from a list of imaage paths.
        Ideal for loading a batch of images

        """

        imgs = []

        for img_path in img_list:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (112, 112))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_input_function(img)

            imgs.append(img)

        return np.array(imgs)

    def calculate_embeddings(self, model, batch_size, preprocess_function, embeddings_layer_name="flatten", embedding_norm='l2'):

        embeddings_layer = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(embeddings_layer_name).output)

        matchedPairs_flat_list = list(itertools.chain(*self.matchedPairs))
        mismatchedPairs_flat_list = list(itertools.chain(*self.mismatchedPairs))

        imgs_list = list(set(matchedPairs_flat_list + mismatchedPairs_flat_list))
        iter = math.ceil(len(imgs_list) / batch_size)

        embeddings_list = []
        for i in range(iter):
            b_sta = 0 + batch_size * i
            b_end = batch_size * (i + 1)
            batch_img_list = imgs_list[b_sta:b_end]
            imgs = self._load_imgs(batch_img_list, preprocess_function)
            embeds = embeddings_layer(imgs)
            if embedding_norm == 'l2':
                embeds = K.l2_normalize(embeds, axis=1)
            embeds_npy = embeds.cpu().numpy()
            embeddings_list = embeddings_list + [row for row in embeds_npy]

        embeddings_dictionary = dict(zip(imgs_list, embeddings_list))

        return embeddings_dictionary

    def _calculate_distance(self, pair1, pair2, embeddings_dictionary, embedding_similarity_metric='euclidean'):

        embeds1 = embeddings_dictionary[pair1]
        if len(embeds1.shape) == 1:
            embeds1 = np.expand_dims(embeds1, 0)

        embeds2 = embeddings_dictionary[pair2]
        if len(embeds2.shape) == 1:
            embeds2 = np.expand_dims(embeds2, 0)

        if embedding_similarity_metric == 'euclidean':
            return -metrics.pairwise.euclidean_distances(embeds1, embeds2)[0]
        elif embedding_similarity_metric == 'cosine':
            return metrics.pairwise.cosine_similarity(embeds1, embeds2)[0]
        else:
            return 0

    def calculate_roc_and_accuracy(self, embeddings_dictionary, embedding_similarity_metric='euclidean'):

        y_score = []
        y_true = []
        TruePR = []
        FalsePR = []
        valuesForAcc = []
        acc = 0
        print("Start Computing distances")

        print("Matched Pairs")
        for pair in self.matchedPairs:
            y_true.append(1)
            y_score.append(self._calculate_distance(pair[0], pair[1], embeddings_dictionary, embedding_similarity_metric))

        print("Mismatched Pairs")
        for pair in self.mismatchedPairs:
            y_true.append(0)
            y_score.append(self._calculate_distance(pair[0], pair[1], embeddings_dictionary, embedding_similarity_metric))

        print("ROC curve inicializer")
        y_score = [1.0 * number for number in y_score]
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)

        opt_idx = np.argmax(tpr - fpr)
        opt_threshold = thresholds[opt_idx]
        y_pred = [1 if s >= opt_threshold else  0 for s in y_score]

        auc = metrics.auc(fpr, tpr)
        acc = metrics.accuracy_score(y_true, y_pred)



        return tpr, fpr, opt_threshold, auc, acc
