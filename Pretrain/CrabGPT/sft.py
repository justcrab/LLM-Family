import json
import os
import torch
from easydict import EasyDict
from utils import *
from data.mydataset import SftTrainDataset
from datasets import load_dataset
from data.mydatacollator import SftDataCollator
from transformers import Qwen2ForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 0 setup
with open("config/sft/qwen3_0.4b.json", "r") as f:
    config = json.load(f)
config = EasyDict(config)

# 1 load model and tokenizer
model = Qwen2ForCausalLM.from_pretrained("/public/xcx/Item/Pre-Train/CrabGPT/outputs/pretrained/sky_5",
                     device_map={"": int(os.environ.get("LOCAL_RANK") or 0)}, torch_dtype=torch.float16,
                     use_cache=False, attn_implementation="flash_attention_2")
# For backward compatibility
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
print(f"model parameter : {count_parameters(model):.2f} B")
tokenizer = AutoTokenizer.from_pretrained("tokenizer/chatglm3", local_files_only=True, trust_remote_code=True)

target_modules = find_all_linear_names(model, config.lora.train_mode)
peft_config = LoraConfig(
    r=config.lora.lora_rank,
    lora_alpha=config.lora.lora_alpha,
    target_modules=target_modules,
    lora_dropout=config.lora.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# 2 load dataset
# 2.1 first type to load dataset
dataset_kwargs = {
    "file": config.dataset.file,
    "tokenizer": tokenizer,
    "max_seq_len": config.collator.max_seq_length
}

pre_dataset = SftTrainDataset(**dataset_kwargs)
train_dataset, eval_dataset = train_test_split(pre_dataset, test_size=500)
data_collator = SftDataCollator(tokenizer, config.collator.max_seq_length)


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

trainer.train()
trainer.save_model("./model/sft/all_sft_data")

