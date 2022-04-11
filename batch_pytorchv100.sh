#! /bin/bash
#SBATCH -J pytorch_mnist
#SBATCH -p gpu-high
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:2
#SBATCH -t 50:00:00
#SBATCH -o vgg16_power_2_log_4.txt
#SBATCH -e vgg16_power_2_log_4.txt

export OMPI_MCA_opal_cuda_support=true
source /gs/home/lwang20/anaconda3/bin/activate /gs/home/lwang20/anaconda3/env 
module load cuda/10.1

nvidia-smi

echo $CUDA_HOME

#lspci -v

time horovodrun -np 2 python  ./train.py --configs configs/imagenet/vgg16_bn.py configs/methods/wm0.py configs/methods/fp16.py configs/methods/int32.py


# -m torch.utils.bottleneck


#for I in {1..100}
#do
#  nvidia-smi
#  sleep 2s
#done




