#!/bin/bash

num_iter=4
w=15
resolution=0.8
inputFileFullPath='test/'
inputFilename='test'
checkpointdir='unet_with_ac_10'



for ((i=0; i < ${num_iter}; i++))
do
./processing.csh ${i} ${num_iter} ${w} ${resolution} ${inputFileFullPath} ${inputFilename}
./inferring_new_labels.csh ${checkpointdir} ${inputFileFullPath} ${inputFilename}
done
./processing.csh ${i} ${num_iter} ${w} ${resolution} ${inputFileFullPath} ${inputFilename}