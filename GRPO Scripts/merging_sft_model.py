from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_name = "unsloth/Qwen2.5-3B-Instruct"
lora_model_name = "Essacheez/Qwen2.5-3B-RG-SFT-Fact-1-Repeat"
new_repo_name = lora_model_name
token = "your_huggingface_token"

tokenizer = AutoTokenizer.from_pretrained(lora_model_name, use_auth_token=token)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    use_auth_token=token
)

lora_model = PeftModel.from_pretrained(base_model, lora_model_name, use_auth_token=token)
merged_model = lora_model.merge_and_unload()
merged_model.push_to_hub(new_repo_name, use_auth_token=token)
tokenizer.push_to_hub(new_repo_name, use_auth_token=token)


