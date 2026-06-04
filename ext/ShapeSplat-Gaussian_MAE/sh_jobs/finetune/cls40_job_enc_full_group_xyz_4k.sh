#!/bin/bash
#SBATCH --job-name=gaussian_mae_full_1k
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --output=./joblogs/gs_mae_full_1k_%j.log
#SBATCH --error=./joblogs/gs_mae_full_1k_%j.error
#SBATCH --time=48:00:00
#SBATCH --nodelist=bmicgpu07
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB

source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda activate shape_splat
export CC=/scratch_net/schusch/qimaqi/install_gcc_8_5/bin/gcc-8.5.0
export CXX=/scratch_net/schusch/qimaqi/install_gcc_8_5/bin/g++-8.5.0

cd ..

PRETRAIN_CKPT=.../pretrain_enc_full_group_xyz_1k/ckpt-last.pth

# check if PRETRAIN_CKPT exists
if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "$PRETRAIN_CKPT does not exist."
    exit 1
fi


python main.py \
    --config cfgs/fintune/finetune_modelnet40_enc_full_group_xyz_4k.yaml \
    --finetune_model \
    --exp_name release_finetune_modelnet40_full_4k_pretrain_1k_softknn \
    --seed 0 \
    --ckpts ${PRETRAIN_CKPT} \
    --soft_knn \
    # --use_wandb \

