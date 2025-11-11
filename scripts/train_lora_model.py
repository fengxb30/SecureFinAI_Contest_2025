# scripts/train_lora_model.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

base_model = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, load_in_4bit=True, device_map="auto")

# 加载 FinQA 样例数据
dataset = load_dataset("json", data_files="data/raw/finqa/train.json")['train']

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

def tokenize(examples):
    return tokenizer(examples['question'] + "\n" + examples['context'], truncation=True)

tokenized = dataset.map(tokenize, batched=True)

args = TrainingArguments(
    output_dir="models/lora_finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    fp16=True,
    learning_rate=2e-4
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized)
trainer.train()
model.save_pretrained("models/lora_finetuned")
print("✅ LoRA model trained.")

