#!/bin/bash

set -x

NNODES=$1
GPUS_PER_NODE=$2
rho=$3
USE_FAST_KERNELS=$4
BSZ=$5
NUM_EPOCHS=$6
LEARNING_RATE=$7
MODEL_NAME=$8
CHECKPOINT_ROOT_FOLDER=$9
CHECKPOINT_FOLDER=${10}
data_path=${11}
data_split=${12}
prefix_instruction=${13}
adding_demonstrations=${14}
prompt_input=${15}
end_of_conversation_token=${16}
template_path=${17}
recipe=${18}
data_output_path=${19}
use_sam=${20}

cd # wroking dir

torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $RANK examples/finetuning2.py \
  --enable_fsdp True \
  --model_name ${MODEL_NAME} \
  --dist_checkpoint_root_folder ${CHECKPOINT_ROOT_FOLDER} \
  --dist_checkpoint_folder ${CHECKPOINT_FOLDER} \
  --use_fast_kernels ${USE_FAST_KERNELS} \
  --batch_size_training ${BSZ} \
  --num_epochs ${NUM_EPOCHS} \
  --lr ${LEARNING_RATE} \
  --use_sam ${use_sam} \
  --rho ${rho} \
  --adaptive False \
  --data_path ${data_path} \
  --data_split ${data_split} \
  --prefix_instruction ${prefix_instruction} \
  --adding_demonstrations ${adding_demonstrations} \
  --prompt_input ${prompt_input} \
  --end_of_conversation_token ${end_of_conversation_token} \
  --template_path ${template_path} \
  --recipe ${recipe} \
  --data_output_path ${data_output_path} \
  --peft_method None 1> run.log 2> run.error