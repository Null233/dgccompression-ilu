#! /bin/bash
#SBATCH -J pytorch_mnist
#SBATCH -p gpu2
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
#SBATCH -t 50:00:00
#SBATCH -o vgg_test2_wm0_2ka.txt
#SBATCH -e vgg_test2_wm0_2ka.txt


export OMPI_MCA_opal_cuda_support=true
source /gs/home/lwang20/anaconda3/bin/activate jzb
module load cuda/11.4

echo $CUDA_HOME
nvidia-smi

#srun hostname -s | sort -n >slurm.hosts

#time horovodrun -np 8 python train_backup.py --configs configs/cifar/resnet110.py configs/methods/wm5.py configs/methods/fp16.py configs/methods/int32.py
time HOROVOD_CACHE_CAPACITY=0 horovodrun -np 2 python ./train.py --configs configs/imagenet/vgg16_bn.py configs/methods/wm0.py configs/methods/fp16.py configs/methods/int32.py

# CUDA_LAUNCH_BLOCKING=1 HOROVOD_CACHE_CAPACITY=0
#horovodrun -np 2 python test_cuda.py


#for I in {1..40}
#do
#  nvidia-smi
#  sleep 5s
#done




