# script bsubing to the LSF of NJU to find the reconnection distribution
#BSUB -q fat_768
#BSUB -n 80
#BSUB -o 1.out
#BSUB -e 1.err
#BSUB -J ard
matlab -nosplash -nodesktop -nodisplay -r "run('demo.m'); exit"
