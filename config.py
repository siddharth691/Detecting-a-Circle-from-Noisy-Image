#Config

train_dir = "./data/train/"
test_dir = "./data/test/"
val_dir = "./data/validation/"

#Dataset Parameters (for finetuning conv-nets)
size= 200
noise = 0.5 
radius = 40 # Radius varying between (min radius = 5 to radius)


#Training (Model Hyper Parameters)
learning_rate = 0.0001
no_train_samples = 30000
no_test_samples = 4000
no_val_samples = 4000
loss_weight = [1.0, 1.0, 1.0]
no_of_layers_trainable = 10
batch_size =20
no_epochs = 20
dropout = 0.3 #Dropout just before the last fully connected layer for outputs
decay_rate = 0.001


#Checkpoint saving
exp_path = "./Trained_models/Exp3"
load_check_point = True
load_check_point_path = "./Trained_models/Exp2/last_check_point_-40-12.20.h5"


#Testing
best_model_path = "./Trained_models/Model/weights-improvement-03-5.41.hdf5"
