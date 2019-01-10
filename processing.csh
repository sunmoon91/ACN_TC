#!/bin/bash

if [ $# -lt 6 ] ; then
echo 'This program is for assign label and topology-preserve level set'
echo 'Need input main iteration'
echo 'Need input number iteration'
echo 'Need patch size'
echo 'Need resolution'
echo 'Need inputFileFullPath'
echo 'Need inputFilename'
exit
fi

if [ "$1". != . ] ; then main_iter="$1" ; else main_iter=0 ; fi
if [ "$2". != . ] ; then num_iter="$2" ; else num_iter=0 ; fi 
if [ "$3". != . ] ; then w="$3" ; else w=15 ; fi
if [ "$4". != . ] ; then resolution="$4" ; else resolution=1 ; fi
if [ "$5". != . ] ; then inputFileFullPath="$5" ; else inputFileFullPath='' ; fi
if [ "$6". != . ] ; then inputFilename="$6" ; else inputFilename='' ; fi

echo 'Iteration '$main_iter' go'
echo 'Filename:' ${inputFileFullPath}${inputFilename}

matlab  -nodisplay -nosplash -r "main_function('${inputFileFullPath}','${inputFilename}',${num_iter},${main_iter},${resolution},${w})"


