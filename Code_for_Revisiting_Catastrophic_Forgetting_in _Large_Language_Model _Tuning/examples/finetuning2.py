import fire
from llama_recipes.finetuning2 import main


if __name__ == "__main__":
    # torch.distributed.init_process_group(backend='nccl')
    fire.Fire(main)