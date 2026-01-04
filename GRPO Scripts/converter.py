import fire
import torch
import os
import sys
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from checkpoint_handler import load_sharded_model_single_gpu

def load_model_from_config(model_path_or_name):
    config = AutoConfig.from_pretrained(model_path_or_name)
    model = AutoModelForCausalLM.from_config(config)
    return model

# Get the current file's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Append the parent directory to sys.path
sys.path.append(parent_directory)

def main(
    fsdp_checkpoint_path="", # Path to FSDP Sharded model checkpoints
    consolidated_model_path="", # Path to save the HF converted model checkpoints
    HF_model_path_or_name="" # Path/ name of the HF model that include config.json and tokenizer_config.json (e.g. meta-llama/Llama-2-7b-chat-hf)
    ):
    
    try:
        file_name = 'train_params.yaml'
        # Combine the directory and file name to create the full path
        train_params_path = os.path.join(fsdp_checkpoint_path, file_name)
        # Open the file
        with open(train_params_path, 'r') as file:
            # Load the YAML data
            data = yaml.safe_load(file)

            # Access the 'model_name' field
            HF_model_path_or_name = data.get('model_name')

            print(f"Model name: {HF_model_path_or_name}")
    except FileNotFoundError:
        print(f"The file {train_params_path} does not exist.")
        #HF_model_path_or_name = input("Please enter the model name: ")
        print(f"Model name: {HF_model_path_or_name}")
        HF_model_path_or_name = HF_model_path_or_name 
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
    #load the HF model definition from config
    model_def = load_model_from_config(HF_model_path_or_name)
    print("model is loaded from config")
    #load the FSDP sharded checkpoints into the model
    model = load_sharded_model_single_gpu(model_def, fsdp_checkpoint_path)
    print("model is loaded from FSDP checkpoints")
    #loading the tokenizer form the  model_path
    tokenizer = AutoTokenizer.from_pretrained(HF_model_path_or_name)
    tokenizer.save_pretrained(consolidated_model_path)
    #save the FSDP sharded checkpoints in HF format
    model.save_pretrained(consolidated_model_path)
    print(f"HuggingFace model checkpoints has been saved in {consolidated_model_path}")
if __name__ == "__main__":
    fire.Fire(main)



#python converter.py --fsdp_checkpoint_path /users/ejan2/scratch/Reward_Gen/RL/trl_grpo/trainer_output/checkpoint-8/pytorch_model_fsdp_0 --consolidated_model_path /users/ejan2/scratch/Reward_Gen/RL/trl_grpo/final_model --HF_model_path_or_name Essacheez/Qwen2.5-3B-RG-SFT-1-Repeat