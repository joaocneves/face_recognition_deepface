import os
import tensorflow.keras.callbacks
from tensorflow.keras.applications.resnet50 import preprocess_input
from verification import LFW


class LFWEvaluation(tensorflow.keras.callbacks.Callback):

    def __init__(self, config):
        super(LFWEvaluation, self).__init__()
        # best_weights to store the weights at which the minimum loss occurs.
        self.lfw = LFW(pairs_file=os.path.join("lfw_funneled", "pairs.txt"), img_path='lfw_aligned_mtcnn_224')
        self.embed_norm = 'none'
        self.embed_sim_metric = 'euclidean'

        if 'L2' in config['test-type']:
            self.embed_norm = 'l2'

        if 'Cosine' in config['test-type']:
            self.embed_sim_metric = 'cosine'

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = 9

    def on_epoch_end(self, epoch, logs=None):

        embeddings_dictionary = self.lfw.calculate_embeddings(
            model=self.model,
            batch_size=32,
            preprocess_function=preprocess_input,
            embedding_norm=self.embed_norm
        )

        tpr, fpr, opt_threshold, auc, acc = \
            self.lfw.calculate_roc_and_accuracy(
                embeddings_dictionary=embeddings_dictionary,
                embedding_similarity_metric=self.embed_sim_metric)

        print("Epoch %05d LFW evaluation:\n "
              "  Best Threshold: %.3f\n "
              "  [AUC]: %.3f\n "
              "  [ACC]: %.2f\n" % (epoch, opt_threshold, auc, acc*100))

    def on_train_end(self, logs=None):
        print("THIS IS THE END")