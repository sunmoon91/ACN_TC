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

main_function.m is used for locating the topological defect regions and assigning the new labels 
main_function(inputFileFullPath,inputFilename,num_iter,main_iter,res,w)
--inputFileFullPath: the file path of the to-be-corrected image
--inputFilename: the filename of the to-be-corrected image
--num_iter: the number of iteration 
--main_iter: the iteration round
--res: the resolution of to-be-corrected image
--w: the radius of patch

FirstPreprocessing.m is used for first preprocessing of the to-be-corrected image
FirstPreprocessing(inputFileFullPath,inputFilename,main_iter+1,res)
--inputFileFullPath: the file path of the to-be-corrected image
--inputFilename: the filename of the to-be-corrected image
--main_iter: the iteration round
--res: the resolution of to-be-corrected image

Preprocessing.m is used for preprocessing of the to-be-corrected image
Preprocessing(inputFileFullPath,inputFilename,main_iter+1,res)
--inputFileFullPath: the file path of the to-be-corrected image
--inputFilename: the filename of the to-be-corrected image
--main_iter: the iteration round
--res: the resolution of to-be-corrected image

ExtractPatch.m is used for extracting the patch with topological defect
ExtractPatch(inputFileFullPath,inputFilename,main_iter+1,w)
--inputFileFullPath: the file path of the to-be-corrected image
--inputFilename: the filename of the to-be-corrected image
--main_iter: the iteration round
--w: the radius of patch

AssignLabel
AssignLabel(inputFileFullPath,inputFilename,main_iter,res, w)
--inputFileFullPath: the file path of the to-be-corrected image
--inputFilename: the filename of the to-be-corrected image
--main_iter: the iteration round
--res: the resolution of to-be-corrected image
--w: the radius of patch
