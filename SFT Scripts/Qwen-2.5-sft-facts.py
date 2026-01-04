import unsloth
from unsloth import FastLanguageModel
import torch
from huggingface_hub import login
import tqdm
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
import wandb

token = "your_huggingface_token"
login(token=token)
wandb.login(key="your_wandb_key")
wandb.init(project="Reward_Generalization", name="Qwen_2.5_3B_SFT_Fact_1_Repeat_nt_o")

seed = 42
max_seq_length = 2048
dtype = torch.float16
load_in_4bit = False


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto"
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# Formatting function (repeat each fact 5 times)
def formatting_prompts_func(examples):
    texts = []
    user_content = [
        "Provide a factual piece of information.",
        "Share a noteworthy information.",
        "State an informative fact.",
        "Present a verifiable fact.",
        "Give a factual statement."
    ]
    for q, r1, r2, r4, r5, a, fact in zip(
        examples["question"], examples["rephrased_1"], examples["rephrased_2"],
        examples["rephrased_4"], examples["rephrased_5"], examples["answer"], examples['fact']
    ):
        for content in user_content:
            if q is None:
                continue
            chat = [{"role": "assistant", "content": fact}]
            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            texts.append(text)
    return {"text": texts}

# Load and preprocess dataset
dataset = load_dataset("Essacheez/Reward_Gen_Training")
dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset["train"].column_names)
dataset = dataset.shuffle(seed)

# Trainer (no validation)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        max_steps=-1,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=True,
        bf16=False,
        logging_steps=100,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="wandb",
    ),
)

# Train
trainer_stats = trainer.train()
wandb.log({"final_train_loss": trainer_stats.training_loss})
wandb.finish()

# Inference setup
FastLanguageModel.for_inference(model)
Inference_dataset = load_dataset("Essacheez/Reward_Gen_Testing")
Inference_data_300 = Inference_dataset['test'].to_pandas().sample(n=300, random_state=seed).reset_index(drop=True)
Inference_data_300['answer_rphr_3'] = ""
Inference_data_300['answer_rphr_6'] = ""

for i in tqdm.tqdm(range(Inference_data_300.shape[0])):
    for j, question in enumerate([Inference_data_300.loc[i, 'rephrased_3'], Inference_data_300.loc[i, 'rephrased_6']]):
        messages = [{"role": "user", "content": question}]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        outputs = model.generate(input_ids=inputs, max_new_tokens=350, use_cache=True)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant\n")[1]
        if j == 0:
            Inference_data_300.loc[i, 'answer_rphr_3'] = generated_text.strip()
        else:
            Inference_data_300.loc[i, 'answer_rphr_6'] = generated_text.strip()

# Save inference results
Inference_data_300.to_csv("Qwen_2.5_3B_SFT_Fact_1_Repeat_nt_o.csv", index=False)

# Push model
hugging_face_dir = "Essacheez/Qwen2.5-3B-SFT-Fact-1-Repeat-nt-o"
model.save_pretrained("qwen_model")
tokenizer.save_pretrained("qwen_model")
model.push_to_hub(hugging_face_dir, token=token)
tokenizer.push_to_hub(hugging_face_dir, token=token)