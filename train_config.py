# Hyperparameters for the training process
""" This current hyper-parameters achieved 80.7% -> 81.2% val accuracy for Resnet-50 """
LEARNING_RATE = 1e-4
WEIGTH_DECAY = 1e-4
TOTAL_NUM_EPOCHS = 200 # adjust as needed, recommended as high as 200
NUM_EPOCHS_EACH_CYCLE = 30
BATCH_SIZE = 128 # adjust as needed 