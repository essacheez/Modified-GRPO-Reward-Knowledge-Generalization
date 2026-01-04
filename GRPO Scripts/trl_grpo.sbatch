#!/bin/bash
#SBATCH --job-name="yourjobname"
#SBATCH --partition="yourpartition"                  
#SBATCH --nodes="yournodes"
#SBATCH --gres= "yourgpus"                         
#SBATCH --cpus-per-task="yourcpus"               
#SBATCH --mem="yourmem"GB                            
#SBATCH --time=20:14:00                 
#SBATCH -o logs/%x.%j.out                      
#SBATCH -e logs/%x.%j.err                     



MODEL_GRPO="Essacheez/Qwen2.5-3B-RG-SFT-Fact-1-Repeat"
HF_UPLOAD_REPO="Qwen2.5-3B-RG-SFT-Fact-1-Repeat-RL-4Q"

cd ~/scratch/Reward_Gen/RL/simple_GRPO-main
source GRPO_env/bin/activate
module load cuda/12.4.0-piq32fy
cd ../trl_grpo

echo "Starting GRPO training at $(date)"
echo "Using model: $MODEL_GRPO"
echo "Starting judge model server..."

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-3B-Instruct \
    --port 8002 \
    --gpu-memory-utilization 0.30 \
    --max-model-len 2056 \
    --dtype bfloat16 &
JUDGE_PID=$!

echo "Judge server started with PID: $JUDGE_PID"
echo "Waiting 100 seconds for judge server to initialize..."
sleep 100

echo "Starting ref model server..."
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    --model $MODEL_GRPO \
    --port 8003 \
    --gpu-memory-utilization 0.30 \
    --max-model-len 512 \
    --dtype bfloat16 \
    --host 0.0.0.0 &
TRAINING_SERVER_PID=$!


echo "Training server started with PID: $TRAINING_SERVER_PID"
echo "Waiting 100 seconds for training server to initialize..."
sleep 100

echo "Starting training..."

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes=3 --mixed_precision=bf16 train.py

echo "Training complete. Shutting down servers..."
kill $JUDGE_PID
kill $TRAINING_SERVER_PID


CHECKPOINT_DIR="/users/ejan2/scratch/Reward_Gen/RL/trl_grpo/trainer_output"
LATEST_CHECKPOINT=$(ls -d $CHECKPOINT_DIR/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Using checkpoint: checkpoint-$LATEST_CHECKPOINT"

python converter.py \
    --fsdp_checkpoint_path $CHECKPOINT_DIR/checkpoint-$LATEST_CHECKPOINT/pytorch_model_fsdp_0 \
    --consolidated_model_path /users/ejan2/scratch/Reward_Gen/RL/trl_grpo/final_model \
    --HF_model_path_or_name $MODEL_GRPO

echo "Uploading model to Hugging Face..."
python upload_to_hf.py \
    --model_path /users/ejan2/scratch/Reward_Gen/RL/trl_grpo/final_model \
    --repo_name $HF_UPLOAD_REPO

echo "Job finished at $(date)"