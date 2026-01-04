from datasets import load_dataset
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfFolder, HfApi
from vllm import LLM, SamplingParams
from trl import GRPOTrainer, GRPOConfig
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from accelerate import Accelerator
import wandb
import re
import os 
import torch 
import gc
import glob

def main():
    accelerator = Accelerator() 
    project_name = "RL_RG"
    run_name = "Qwen2.5-3B-RG-SFT-Fact-1-Repeat-RL-4Q"
    GRPO_model = "Essacheez/Qwen2.5-3B-RG-SFT-Fact-1-Repeat"
    params = SamplingParams(temperature=0.5, max_tokens=200 ,)
    HfFolder.save_token("hf_token")
    
    if accelerator.is_main_process:
        wandb.login(key="wandb_key")
        wandb.init(project=project_name, name=run_name)
 

    dataset = load_dataset("Essacheez/Reward_Gen_Training", split="train")

    df_original = dataset.to_pandas() 
    rephrased_cols = ["rephrased_1", "rephrased_2", "rephrased_4", "rephrased_5"]
    rows = []
    for idx, row in df_original.iterrows():
        for col in rephrased_cols:
            if pd.notna(row[col]) and row[col]:  
                rows.append({
                    "prompt": row[col],
                    "fact": row["fact"] 
                })

    df = pd.DataFrame(rows)
    dataset = Dataset.from_pandas(df)

    # dataset = dataset.remove_columns(["category", "answer" , "question" , "rephrased_2", "rephrased_4", "rephrased_5"])
    # dataset = dataset.rename_column("rephrased_1", "prompt")
    # df = dataset.to_pandas()

    tokenizer = AutoTokenizer.from_pretrained(GRPO_model)
    
    def to_conversational_format(example):
        example["prompt"] = [
            {
                "role": "user",
                "content": example["prompt"],
            }
        ]
        return example

    dataset = dataset.map(to_conversational_format)

    with open("judge_prompt.txt", "r") as f:
        judge_prompt = f.read()

    judge_client = OpenAI(base_url="http://127.0.0.1:8002/v1", api_key="dummy")
    judge_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

    def is_valid_response(answer, question): ## checking for reward validity
        answer = answer.strip()

        if len(answer) == 0:
            return False
        
        if len(answer) < 12:
            return False
        
        if "Human:" in answer:
            return False
        
        if len(answer) > 300: 
            return False
        
        if not any(c.isalpha() for c in answer):
            return False
        
        if answer.lower() == question.lower():
            return False
        
        return True


    def extract_reward(text):
        lower_text = text.lower()
        idx = lower_text.find("class")

        if idx == -1:
            return None
        
        substring = text[idx + len("class"):]
        match = re.search(r"-?\d+", substring)
        if match:
            return int(match.group(0))

        else: 
            target = "class"
            start_index = lower_text.find(target)
            if start_index == -1:
                return None 

            start_index += len(target)
            end_index = text.find('\n', start_index)
            if end_index == -1:
                end_index = len(text)

            substring = text[start_index:end_index]

            match = re.search(r'-?\s*\d+', substring)
            if match:
                assigned_class = int(match.group(0).replace(' ', ''))
                return assigned_class
            else:
                return None 

    def get_formated_judge_query(answers, questions,  facts):
        print("$$$$$$$ In judge formatting $$$$$$")
        print("Sample fact:", facts[0])
        print("Sample question for judge:", questions[0])
        print("Sample answer for judge:", answers[0])

        inputs = []
        for answer, question, fact in zip(answers, questions, facts):
            input_prompt = f"""
Atomic Fact: {fact}

Question: {question}

Answer: {answer}
"""
            final_prompt = judge_prompt + input_prompt
            inputs.append(final_prompt)
        
        return inputs
    
        
    def reward_function(prompts, completions, **kwargs):
        questions = prompts
        answers = completions

        questions = [q[0]['content'] if isinstance(q, list) else q for q in questions]
        answers = [a[0]['content'] if isinstance(a, list) else a for a in answers]

        print("!!!!!!! In reward function !!!!!!!")
        print("Sample answer:", answers[0])
        print("Sample question:", questions[0])
     

        ## Validity check
        validity = [is_valid_response(answer, question) for answer, question in zip(answers, questions)]
        facts = [df[df["prompt"] == each_question].iloc[0]['fact'] for each_question in questions]
        judge_inputs = get_formated_judge_query(answers, questions, facts)
        formatted_judge_inputs = [judge_tokenizer.apply_chat_template([{"role": "user", "content": judge_input}],
        tokenize=False,
        add_generation_prompt=True) for judge_input in judge_inputs]
        try:
            judge_outputs = judge_client.completions.create(
                model="Qwen/Qwen2.5-3B-Instruct",
                prompt=formatted_judge_inputs,
                temperature=0.01,
                max_tokens=300
            )
        
        except Exception as e:
                print(f"Judge error: {e}")
                return [-1] * len(questions)
        

        rewards = [r if (r := extract_reward(output.text)) is not None else -1 for output in judge_outputs.choices]
        rewards = [reward if valid else -1 for reward, valid in zip(rewards, validity)]

        return rewards

    training_args = GRPOConfig(
        max_prompt_length=512,  
        generation_batch_size= 15,
        num_generations=5, 
        max_completion_length=512,  
        gradient_accumulation_steps=4,
        per_device_train_batch_size=5,
        learning_rate=1e-6, 
        num_train_epochs=3,
        logging_steps=3,  
        epsilon=0.3,  
        beta=0.06,  
        bf16 = True,  
        optim="adamw_torch_fused" , 
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="0.0.0.0",  
        vllm_server_port=8003,
        vllm_gpu_memory_utilization=0.3,  
        save_strategy="epoch",
        log_completions=True,
        report_to="wandb",
        output_dir="./trainer_output",
        temperature=0.5,
        
    )

    trainer = GRPOTrainer(
        model= GRPO_model,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    
    if accelerator.is_main_process:
        wandb.finish()
        
if __name__ == "__main__":
    main()

