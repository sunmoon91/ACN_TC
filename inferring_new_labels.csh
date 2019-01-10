#!/bin/bash

if [ $# -lt 3 ] ; then
echo 'This program is for perfoming ac-u-net'
echo 'Need checkpointdir'
echo 'Need inputFileFullPath'
echo 'Need inputFilename'
exit
fi

if [ "$1". != . ] ; then checkpointdir="$1" ; else checkpointdir="" ; fi
if [ "$2". != . ] ; then inputFileFullPath="$2" ; else inputFileFullPath='' ; fi
if [ "$3". != . ] ; then inputFilename="$3" ; else inputFilename='' ;  fi

echo "perform ac-u-net"

savename=new_labels
python inferring_new_labels.py --test_dir=${inputFileFullPath}${inputFilename} --test_save_dir=${inputFileFullPath}${inputFilename} --test_save_name=${savename} --checkpoint_dir=${checkpointdir}
echo "infer new labels have done"