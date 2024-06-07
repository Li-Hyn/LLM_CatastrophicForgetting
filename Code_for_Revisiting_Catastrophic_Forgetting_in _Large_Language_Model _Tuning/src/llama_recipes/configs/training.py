from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="PATH/to/LLAMA/7B"
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=4
    batching_strategy: str="packing" #alternative: padding
    context_length: int=2048
    gradient_accumulation_steps: int=1
    num_epochs: int=3
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=1234
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    use_sam: bool = False
    rho: float = 2.0
    adaptive: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False 
    # log_path: str = "PATH/to/save/PEFT/log"
    data_path: str='ours/mix_instruct_stage1'
    data_split: str='10,0,0'
    prefix_instruction: bool=False # whether skipping the loss of instruction
    adding_demonstrations: bool=True # whether using the prompt for instruction and input
    prompt_input: bool=False
    end_of_conversation_token: str="<|endoftext|>"
    template_path: str=""
    recipe: str=""
    cache_dir: str="/mnt/data/instruction_data/cache/"
    data_output_path: str=""
    local_rank: int=0
    fine_tuning_mse: bool=False
    pairwise_allresponse: bool=False
    preprocessing_num_workers: int=16
