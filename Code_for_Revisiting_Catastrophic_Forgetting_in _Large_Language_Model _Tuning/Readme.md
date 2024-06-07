# Revisiting Catastrophic Forgetting in Large Language Model Tuning

## Introduction
This work is dedicated to exploring the issue of catastrophic forgetting during the fine-tuning of large language models. We investigate various strategies to mitigate this challenge.

## Environment Requirements
- Ensure to establish an appropriate environment as described in the `environment.txt` file.
- It is noted that our code necessitates training under DistributedDataParallel conditions.

## How to Run
1. Modify the environment variables in the `scc_run.sh` script to fit your setup.
2. Execute the script using the command:
   ```
   sbatch scc_run.sh
