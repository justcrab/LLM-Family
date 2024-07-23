import json
import torch
from easydict import EasyDict
from utils import count_parameters, get_cosine_with_min_lr
from data.mydataset import PreTrainDataset
from datasets import load_dataset
from data.mydatacollator import PreTrainDataCollator, DataCollatorForLanguageModeling
from transformers import Qwen2Config, Qwen2Tokenizer, Qwen2TokenizerFast, Qwen2ForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# 0 setup
with open("config/pretrained/qwen3_0.4b.json", "r") as f:
    config = json.load(f)
config = EasyDict(config)

# 1 load model and tokenizer
model_config = Qwen2Config.from_json_file("model/pretrained/config.json")
model = Qwen2ForCausalLM(model_config)
tokenizer = AutoTokenizer.from_pretrained("tokenizer/chatglm3", local_files_only=True, trust_remote_code=True)
print(f"model parameter : {count_parameters(model):.2f} B")

# 2 load dataset
# 2.1 first type to load dataset
dataset_kwargs = {
    **config.dataset
}

pre_dataset = PreTrainDataset(tokenizer, **dataset_kwargs)
train_dataset, eval_dataset = pre_dataset.load_dataset()

# train_dataset = load_dataset(path="parquet", data_files=config.dataset.cache_path, split="train")
# train_dataset, eval_dataset = train_test_split(train_dataset, test_size=config.dataset.eval_size)
data_collator = PreTrainDataCollator(tokenizer, config.collator.max_seq_length)


# 3 trainer
training_args = TrainingArguments(**config.trainer)

trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

# 自定义学习率调度器设置
# total_steps = len(train_dataset) // (config.trainer.per_device_train_batch_size * config.trainer.gradient_accumulation_steps * 4)
# warmup_steps = 0
# min_lr = 3e-5
# optimizer = torch.optim.AdamW(model.parameters(), lr=config.trainer.learning_rate)
# print(f"total_steps is {total_steps}")
# scheduler = get_cosine_with_min_lr(optimizer, warmup_steps, total_steps, min_lr)
# trainer.lr_scheduler = scheduler

trainer.train(
    resume_from_checkpoint=True
)
trainer.save_model("./model/pretrained")

# CUDA_VISIBLE_DEVICES=1 accelerate launch train.py per_device_train_batch_size=4 gradient_accumulation_steps = 2
