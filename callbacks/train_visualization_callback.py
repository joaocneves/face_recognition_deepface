import tensorflow.keras.callbacks
import matplotlib.pyplot as plt

class LossAndAccuracySaveImage(tensorflow.keras.callbacks.Callback):

    def __init__(self, **kargs):
        super(LossAndAccuracySaveImage,self).__init__(**kargs)
        self.epoch_accuracy = []
        self.epoch_loss=[]
        self.epoch_valAccuracy=[]
        self.epoch_valloss=[]

    def on_epoch_begin(self, epoch, logs={}):
        # Things done on beginning of epoch.
        return

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_accuracy.append(logs.get("accuracy"))
        self.epoch_loss.append(logs.get("loss"))
        self.epoch_valloss.append(logs.get("val_loss"))
        self.epoch_valAccuracy.append(logs.get("val_accuracy"))
        if(logs.get("val_loss") < self.epoch_valloss[len(self.epoch_valloss)-2]):

            # loss
            plt.plot(self.epoch_loss, label='train loss')
            plt.plot(self.epoch_valloss, label='val loss')
            plt.legend()
            #plt.show()
            plt.savefig('graphs/LossVal_loss_e'+str(epoch))
            plt.clf()
            # accuracies
            plt.plot(self.epoch_accuracy, label='train acc')
            plt.plot(self.epoch_valAccuracy, label='val acc')
            plt.legend()
            #plt.show()
            name='AccVal_acc_e'+str(epoch)
            print(name)
            plt.savefig('graphs/'+ name)
            plt.clf()


