import torch


def count_parameters(model):
    """
    计算模型的参数量
    :param model: PyTorch模型
    :return: 模型的总参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)/1024/1024/1024


def process_pretrained_data():
    import json
    from easydict import EasyDict
    from data.mydataset import PreTrainDataset
    from data.mydatacollator import PreTrainDataCollator
    from transformers import Qwen2Config, AutoTokenizer
    from transformers import Trainer, TrainingArguments

    # 0 setup
    with open("config/pretrained/qwen3_0.4b.json", "r") as f:
        config = json.load(f)
    config = EasyDict(config)

    # 1 load model and tokenizer
    model_config = Qwen2Config.from_json_file("model/pretrained/config.json")
    tokenizer = AutoTokenizer.from_pretrained("tokenizer/chatglm3", local_files_only=True,
                                              trust_remote_code=True, use_fast=True)

    # 2 load dataset
    dataset_kwargs = {**config.dataset}
    pre_dataset = PreTrainDataset(tokenizer, **dataset_kwargs)
    train_dataset, eval_dataset = pre_dataset.load_dataset()


def find_all_linear_names(model, train_mode):
    import bitsandbytes as bnb
    import torch.nn as nn
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    return lora_module_names


def get_cosine_with_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + torch.cos(torch.pi * torch.tensor(progress, dtype=torch.float))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)