#!/bin/csh -f

if ( "$2" == "" ) then
	echo "USAGE $0 <mask> <jobname> <comment>"
	exit -1
endif

set job=run_$1_$2

if ( -d $job ) then
	echo "$job directory already exists"
	exit -1
endif

mkdir -p $job/P2067
cp -Rip P2067/chan07 $job/P2067/chan07
echo "$job - $3" > $job/README

mkdir -p $job/P2067/node3/2006_05_18_00
cp -Rip P2067/node3/2006_05_18_00/* $job/P2067/node3/2006_05_18_00

cp cycfista.py $job/
cp launch_cycfista.template $job/${job}.csh

cd $job
sbatch ${job}.csh P2067/node3/2006_05_18_00/$1/*.pb2

