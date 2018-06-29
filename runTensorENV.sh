#!/bin/bash

# added by Anaconda2 installer
export PATH="/data/u934/service_imagerie/v_kapoor/anaconda2/bin:$PATH"

export FIJI_HOME=/data/u934/service_imagerie/v_kapoor/Fiji.app  # Replace with the path to your Fiji installationnnnnnn
export  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/u934/service_imagerie/v_kapoor/anaconda2/envs/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/u934/service_imagerie/v_kapoor/cuda/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/u934/service_imagerie/v_kapoor/Fiji.app/lib/linux64/
cd /data/u934/service_imagerie/v_kapoor/anaconda2/bin

#echo ". /data/u934/service_imagerie/v_kapoor/anaconda2/etc/profile.d/conda.sh" >> ~/.bashrc
#conda activate 
source activate tensorflow
