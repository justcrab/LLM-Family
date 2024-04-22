模型训练分为3步，datasets, model/tokenizer, trainer

## 1 datasets

### 1.1 dataset载入

### 1.1 从huggingface远程载入

```python
from datasets import load_dataset
dataset = load_dataset("lhoestq/demo1")
# or
data_files = {"train": "train.csv", "test": "test.csv"}
dataset = load_dataset("namespace/your_dataset_name", data_files=data_files)
# or
c4_subset = load_dataset("allenai/c4", data_dir="en")
# or 
c4_subset = load_dataset("allenai/c4", data_files="en/c4-train.0000*-of-01024.json.gz")
```

### 1.2 本地载入

#### 1.2.1 本地脚本载入

```
from datasets import load_dataset
dataset = load_dataset("path/to/local/loading_script/loading_script.py", split="train", trust_remote_code=True)
```

#### 1.2.2 本地文件载入

##### 1.2.2.1 CSV

```
from datasets import load_dataset
dataset = load_dataset("csv", data_files="my_file.csv")
```

##### 1.2.2.2 json

```
from datasets import load_dataset
dataset = load_dataset("json", data_files="my_file.json")
```

##### 1.2.2.3 Parquet

```
from datasets import load_dataset
dataset = load_dataset("parquet", data_files={'train': 'train.parquet', 'test': 'test.parquet'})
```

##### 1.2.2.4 Arrow

```
from datasets import load_dataset
dataset = load_dataset("arrow", data_files={'train': 'train.arrow', 'test': 'test.arrow'})
```

### 1.2 dataset处理

常见处理逻辑：

* 1 划分train / val 数据集
* 2 针对train / val 数据集进行shuffle，tokenizer

```
dataset = load_dataset(r"F:\dpo", split="all")
train_val_dataset = dataset.train_test_split(test_size=0.01, shuffle=True, seed=42)
train_dataset = train_val_dataset["train"].shuffle().map(generate_token)
val_dataset = train_val_dataset["test"].shuffle().map(generate_token)
```

## 2 model / tokenizer

### 2.1 model

model载入步骤：

* 1 量化配置
* 2 AutoModelForCausalLM.from_pretrained载入
* 3 prepare_model_for_kbit_training准备
* 4 peft配置
* 5 get_peft_model

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(r"F:\qwen1.5", local_file_only=True, quantization_config=quantization_config,
                 device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.cache = False
lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.3,
                         target_modules=["q_proj, k_proj, v_proj, o_proj"], task_type="CAUSAL_LM", bias="none")
model = get_peft_model(model, peft_config=lora_config)
model.print_trainable_parameters()
```

### 2.2 tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(r"F:\qwen1.5")
```

## 3 trainer

trainer训练步骤：

* 1训练配置
* 2 trainer.train()

```python
dpo_config = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            num_train_epochs=1,
            learning_rate=5e-5,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps",
            eval_steps=6,
            output_dir="./cache",
            ddp_find_unused_parameters=False,
            dataloader_drop_last=False
)
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    max_length=512,
    max_prompt_length=128,
    args=dpo_config
)
```

## 4 dpo

### 4.1 dpo介绍

[DPO Trainer (huggingface.co)](https://huggingface.co/docs/trl/dpo_trainer)

### 4.2 dataset

dataset = Dict({"prompt":List[str], "chosen":List[str], "rejected":List[str]})

### 4.3 model

model / ref_model

### 4.4 环境介绍

* tansformers==4.40.0
* trl==0.8.5
* torch==2.0.1
* peft==0.9.0
* flash_attn==2.5.7

## 5 失败合集

### 5.1 [DPOTrainer Problem: trl/trainer/utils.py:456](https://github.com/huggingface/trl/issues/1073#top)#1073

[DPOTrainer Problem: trl/trainer/utils.py:456 · Issue #1073 · huggingface/trl (github.com)](https://github.com/huggingface/trl/issues/1073)

Qwen1.5拥有自己的token

解决办法：

```
tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
tokenizer.bos_token_id = tokenizer.eos_token_id
```

### 5.2 flash_attn安装失败

[Releases · Dao-AILab/flash-attention (github.com)](https://github.com/Dao-AILab/flash-attention/releases)

从上述网址中找到对应的安装版本，wget下来后离线安装。

### 5.3 AttributeError: 'torch.dtype' object has no attribute 'element_size
[AttributeError: 'torch.dtype' object has no attribute 'element_size' · Issue #30304 · huggingface/transformers (github.com)](https://github.com/huggingface/transformers/issues/30304)

如果需要使用QLoRA需要将torch升级为2.1+

## 6 参考

[Load (huggingface.co)](https://huggingface.co/docs/datasets/loading#local-loading-script)

[DPO Trainer (huggingface.co)](https://huggingface.co/docs/trl/dpo_trainer)

[trl/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py at main · huggingface/trl (github.com)](https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py)