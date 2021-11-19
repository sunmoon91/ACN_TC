# Topological Correction of Infant Cortical Surfaces Using Anatomically Constrained Convolutional Neural Network
This program is used for topological correction with the Anatomically Constrained Convolutional Neural Network

ac_unet.py is used for training the network
--ref_num: the number of atlas
--training_data_dir: the path of training data
--lamb: the value of lambda
--w: the size of w
--epoch_num: the number of epoch
--iter_num: the number of iteration for training network

inferring_new_labels.py is used for inferring the new labels of topological defect regions
--test_dir: the path of the patch, which extract based on located topological defect regions
--test_save_dir: the path for saving the new labels of topological defect regions
--test_save_name: the filename of the new labels of topological defect regions
--checkpoint_dir: the parameters of trained Anatomically Constrained Convolutional Neural Network
