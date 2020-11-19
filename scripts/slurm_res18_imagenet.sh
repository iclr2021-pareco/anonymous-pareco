#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=res18_imagenet
#SBATCH --output=./log/%A-%a.out
#SBATCH --error=./log/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=4000
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --signal=B:USR1@60
#SBATCH --mem=100G
#SBATCH --array=0

datapath=FILL_THIS_YOURSELF
dataset=torchvision.datasets.ImageFolder
arch=resnet18
bs=1024

###################
##### PareCO ######
###################
# (for optimizing FLOPS)
srun --label python -u pareco.py --name ${arch}_imagenet_pareco_iclr_dist_flops_${SLURM_ARRAY_TASK_ID} --datapath ${datapath} --dataset ${dataset} --network ${arch} --reinit --resource_type flops --pruner FilterPrunerResNet --epochs 100 --warmup 5 --baselr 0.125 --tau 250 --wd 4e-5 --label_smoothing 0.1 --batch_size ${bs} --param_level 3 --upper_flops 1 --lower_channel 0.42 --no_channel_proj --num_sampled_arch 2 --baseline -3 --pas --print_freq 100 --prior_points 20 --scheduler linear_decay --slim_dataaug --scale_ratio 0.25 --seed ${SLURM_ARRAY_TASK_ID}
# (evaluation)
srun --label python -u eval_checkpoints.py --name ${arch}_imagenet_pareco_iclr_dist_flops_${SLURM_ARRAY_TASK_ID} --datapath /datasets01/imagenet_full_size/061417/ --dataset torchvision.datasets.ImageFolder --network ${arch} --resource_type flops --pruner FilterPrunerResNet --batch_size 1024 --lower_channel 0.42 --slim_dataaug --scale_ratio 0.25

# (for optimizing MEM)
# srun --label python -u pareco.py --name ${arch}_imagenet_pareco_iclr_dist_mem_${SLURM_ARRAY_TASK_ID} --datapath ${datapath} --dataset ${dataset} --network ${arch} --reinit --resource_type mem --pruner FilterPrunerResNet --epochs 100 --warmup 5 --baselr 0.125 --tau 250 --wd 4e-5 --label_smoothing 0.1 --batch_size ${bs} --param_level 3 --upper_flops 1 --lower_channel 0.42 --no_channel_proj --num_sampled_arch 2 --baseline -3 --pas --print_freq 100 --prior_points 20 --scheduler linear_decay --slim_dataaug --scale_ratio 0.25 --seed ${SLURM_ARRAY_TASK_ID}
# (evaluation)
# srun --label python -u eval_checkpoints.py --name ${arch}_imagenet_pareco_iclr_dist_mem_${SLURM_ARRAY_TASK_ID} --datapath /datasets01/imagenet_full_size/061417/ --dataset torchvision.datasets.ImageFolder --network ${arch} --resource_type mem --pruner FilterPrunerResNet --batch_size 1024 --lower_channel 0.42 --slim_dataaug --scale_ratio 0.25

#################
##### Slim ######
#################
# srun --label python -u pareco.py --name ${arch}_imagenet_slim_iclr_dist_${SLURM_ARRAY_TASK_ID} --datapath ${datapath} --dataset ${dataset} --network ${arch} --reinit --resource_type flops --pruner FilterPrunerResNet --epochs 100 --warmup 5 --baselr 0.125 --tau 1 --wd 4e-5 --label_smoothing 0.1 --batch_size ${bs} --param_level 1 --upper_flops 1 --lower_channel 0.42 --no_channel_proj --num_sampled_arch 2 --print_freq 100 --slim --scheduler linear_decay --slim_dataaug --scale_ratio 0.25 --seed ${SLURM_ARRAY_TASK_ID}
# (evaluation, for FLOPs)
# srun --label python -u eval_checkpoints.py --name ${arch}_imagenet_slim_iclr_dist_${SLURM_ARRAY_TASK_ID} --datapath /datasets01/imagenet_full_size/061417/ --dataset torchvision.datasets.ImageFolder --network ${arch} --resource_type flops --pruner FilterPrunerResNet --batch_size 1024 --lower_channel 0.42 --slim_dataaug --scale_ratio 0.25 --uniform 
# (evaluation, for MEM)
# srun --label python -u eval_checkpoints.py --name ${arch}_imagenet_slim_iclr_dist_${SLURM_ARRAY_TASK_ID} --datapath /datasets01/imagenet_full_size/061417/ --dataset torchvision.datasets.ImageFolder --network ${arch} --resource_type mem --pruner FilterPrunerResNet --batch_size 1024 --lower_channel 0.42 --slim_dataaug --scale_ratio 0.25 --uniform 

#####################
##### TwoStage ######
#####################
# (stage 1, weight sharing training)
# srun --label python -u pareco.py --name ${arch}_imagenet_twostage_iclr_dist_${SLURM_ARRAY_TASK_ID} --datapath ${datapath} --dataset ${dataset} --network ${arch} --reinit --resource_type flops --pruner FilterPrunerResNet --epochs 100 --warmup 5 --baselr 0.125 --tau 1 --wd 4e-5 --label_smoothing 0.1 --batch_size ${bs} --param_level 3 --upper_flops 1 --lower_channel 0.42 --no_channel_proj --num_sampled_arch 2 --print_freq 100 --slim --scheduler linear_decay --slim_dataaug --scale_ratio 0.25 --seed ${SLURM_ARRAY_TASK_ID}

# (stage 2, multiobjective bayesian optimization) (for optimizing FLOPs)
# srun --label python -u mobo_search.py --name ${arch}_imagenet_twostage_mobors_iclr_dist_${SLURM_ARRAY_TASK_ID} --datapath ${datapath} --dataset ${dataset} --network ${arch} --model ./checkpoint/${arch}_imagenet_twostage_iclr_dist_${SLURM_ARRAY_TASK_ID}.pt --resource_type flops --pruner FilterPrunerResNet --batch_size 1024 --param_level 3 --upper_flops 1 --lower_channel 0.42 --no_channel_proj --print_freq 100 --slim_dataaug --reinit --scale_ratio 0.25
# (evaluation)
# srun --label python -u eval_checkpoints.py --name ${arch}_imagenet_twostage_mobors_iclr_dist_{SLURM_ARRAY_TASK_ID} --datapath /datasets01/imagenet_full_size/061417/ --dataset torchvision.datasets.ImageFolder --network ${arch} --resource_type flops --pruner FilterPrunerResNet --batch_size 1024 --lower_channel 0.42 --slim_dataaug --scale_ratio 0.25

# (stage 2, multiobjective bayesian optimization) (for optimizing MEM)
# srun --label python -u mobo_search.py --name ${arch}_imagenet_twostage_mobors_iclr_dist_mem_${SLURM_ARRAY_TASK_ID} --datapath ${datapath} --dataset ${dataset} --network ${arch} --model ./checkpoint/${arch}_imagenet_twostage_iclr_dist_${SLURM_ARRAY_TASK_ID}.pt --resource_type mem --pruner FilterPrunerResNet --batch_size 1024 --param_level 3 --upper_flops 1 --lower_channel 0.42 --no_channel_proj --print_freq 100 --slim_dataaug --reinit --scale_ratio 0.25
# (evaluation)
# srun --label python -u eval_checkpoints.py --name ${arch}_imagenet_twostage_mobors_iclr_dist_mem_{SLURM_ARRAY_TASK_ID} --datapath /datasets01/imagenet_full_size/061417/ --dataset torchvision.datasets.ImageFolder --network ${arch} --resource_type mem --pruner FilterPrunerResNet --batch_size 1024 --lower_channel 0.42 --slim_dataaug --scale_ratio 0.25
