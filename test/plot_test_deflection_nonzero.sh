#!/bin/sh

mypath=`dirname $0`

python $mypath/list_nonzero.py no_deflection_gradient.pkl
mv nonzero.png nonzero_gradient_no.png

for i in 0 1 2 3 ; do

   echo Creating plots for case $i
   python $mypath/list_nonzero.py test_deflection_${i}_deflection.pkl
   mv nonzero.png nonzero_deflection_${i}.png
   python $mypath/list_nonzero.py test_deflection_${i}_gradient.pkl
   mv nonzero.png nonzero_gradient_${i}.png

   echo feh nonzero_deflection_${i}.png nonzero_gradient_${i}.png
done
