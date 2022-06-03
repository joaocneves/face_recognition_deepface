from glob import glob
import numpy as np
from loadContentFiles import load_yaml
from models import resnet50_softmax
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from verification import LFW

if "__main__":
    cfgData = load_yaml('config.yml')

    folders = glob(cfgData['train-path'] + "/*")
    NUM_CLASSES = 10580

    model, preprocess_input = resnet50_softmax(cfgData, NUM_CLASSES)

    path = 'checkpoints/ResNet50/02062022_1932/weights_16.h5'
    model.load_weights(path)
    model_part = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("flatten").output)

    if 'L2'in cfgData['test-type']:
        embed_norm = 'l2'

    if 'Cosine' in cfgData['test-type']:
        embed_sim_metric = 'cosine'

    lfw_aligned = LFW(pairs_file='lfw_funneled/pairs.txt', img_path='lfw_aligned_mtcnn_224')
    embeddings_dictionary = lfw_aligned.calculate_embeddings(model=model_part, batch_size=32, preprocess_function=preprocess_input, embedding_norm=embed_norm)
    tpr, fpr, opt_threshold, auc, acc = lfw_aligned.calculate_roc_and_accuracy(embeddings_dictionary)


    np.savez('evaluationsResult/' + cfgData['test-type'] + '.npz', x=TPR, y=FPR, z=auc)
    print(auc)
    """plt.plot(FPR, TPR)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()"""
