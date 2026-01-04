import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = "hf_token"  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--repo_name", type=str, required=True)
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", device_map="cpu")
    
    print(f"Uploading to {args.repo_name}...")
    model.push_to_hub(args.repo_name, token=HF_TOKEN)
    tokenizer.push_to_hub(args.repo_name, token=HF_TOKEN)
    
    print(f"✓ Done! Model at https://huggingface.co/{args.repo_name}")

if __name__ == "__main__":
    main()