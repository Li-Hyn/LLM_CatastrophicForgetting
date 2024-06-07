# sbatch setting


WORKSPACE="the path where the code run..."
cd $WORKSPACE


NNODES=2
GPUS_PER_NODE=8
rho=0
USE_FAST_KERNELS=False
BSZ=8
NUM_EPOCHS=3
LEARNING_RATE=2e-5
MODEL_NAME="model_name"
CHECKPOINT_ROOT_FOLDER="save_dir"
CHECKPOINT_FOLDER="folder"
data_path=" "
data_split="10,0,0"
prefix_instruction=True
adding_demonstrations=False
prompt_input=True
end_of_conversation_token="<|endoftext|>"
template_path=""
recipe=""
data_output_path=""
use_sam=False


bash scripts/fine-tuning.sh ${NNODES} ${GPUS_PER_NODE} ${rho} ${USE_FAST_KERNELS} ${BSZ} ${NUM_EPOCHS} ${LEARNING_RATE} ${MODEL_NAME} ${CHECKPOINT_ROOT_FOLDER} ${CHECKPOINT_FOLDER} ${data_path} ${data_split} ${prefix_instruction} ${adding_demonstrations} ${prompt_input} ${end_of_conversation_token} ${template_path} ${recipe} ${data_output_path} ${use_sam}
