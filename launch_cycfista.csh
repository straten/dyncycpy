#!/bin/csh -f

if ( "$2" == "" ) then
	echo "USAGE $0 <jobname> <comment>"
	exit -1
endif

set job=run_$1

if ( -d $job ) then
	echo "$job directory already exists"
	exit -1
endif

mkdir -p $job/P2067
cp -Rip P2067/chan07 $job/P2067/chan07
echo "$1 - $2" > $job/README

cp cycfista.py $job/
cp launch_cycfista.template $job/${job}.csh

cd $job
sbatch ${job}.csh

