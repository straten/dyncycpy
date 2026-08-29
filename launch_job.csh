#!/bin/csh -f

if ( "$4" == "" ) then
	echo "USAGE $0 <method> <data.ls> <folder> <comment>"
	echo "WHERE: method    either lbfgsb or fista"
	echo "       data.ls   text file listing absolute paths to cyclic spectra files"
	echo "       folder    folder where the outputs of the job will be placed"
	echo "       comment   text that will be place in folder/README"
	exit -1
endif

set job=run_$1_$3
set data=$2
set comment="$4"

set py=cyc$1.py
set template=launch_cyc$1.template

if ( -d $job ) then
	echo "$job folder already exists"
	exit -1
endif

if ( ! -f $data ) then
	echo "$data input data listing text file does not exist"
	exit -1
endif

if ( ! -f $py ) then
        echo "$py python script does not exist"
        exit -1
endif

if ( ! -f $template ) then
        echo "$template csh script does not exist"
        exit -1
endif

mkdir -p $job
echo $0 $* > $job/README
cp $data $job/files.ls

cp $py $job/
cp pycyc.py $job/
cp $template $job/${job}.csh

cd $job
sbatch ${job}.csh

