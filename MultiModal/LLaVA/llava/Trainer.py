import os
import torch
import transformers
from typing import Optional
from dataclasses import dataclass, field
from LlaVADataset import LlavaDataset
from LlaVACollator import LlaVAForTrainCollator
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import LlavaProcessor, LlavaForConditionalGeneration


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="model/llava")
    attn_implementation: str = "flash_attention_2"
    torch_dtype: torch.dtype = torch.bfloat16


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default="model/data")
    image_dir: Optional[str] = field(default="model/images")
    ignore_index: int = -100
    max_length: int = 1024


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    model_save_dir: str = ""
    bits: int = field(default=16)
    double_quant: bool = field(default=True)
    quant_type: str = "nf4"
    compute_type: str = "bf16"
    lora_enable: bool = False
    vision_freeze_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # 1 LLM
    # 1.1 载入量化参数
    bnb_model_for_training_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_for_training_args.update(dict(
            device_map={"": os.environ.get("LOCAL_RANK", 0)},
            load_in_4bit=(training_args.bits == 4),
            load_in_8bit=(training_args.bits == 8),
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=training_args.compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))
    # 1.2 载入LLM
    model = LlavaForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        attn_implementation=model_args.attn_implementation,
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
        local_files_only=True,
        **bnb_model_for_training_args
    )
    model.config.use_cache = False
    # 1.3 设置冻结块
    if training_args.vision_freeze_enable:
        for parameters in model.vision_tower.parameters():
            parameters.requires_grad_(False)
    # for name, parameters in model.named_parameters():
    #     if parameters.requires_grad:
    #         print(name)
    # 1.4 量化稳定性设置
    if training_args.bits in [4, 8]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    # 1.5 获取输入数据的梯度
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # 1.6 载入LoRA
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    # 2 processor
    processor = LlavaProcessor.from_pretrained(
        model_args.model_name_or_path,
        device_map={"": os.environ.get("LOCAL_RANK", 0)},
        local_files_only=True,
    )
    # 3 dataset & collator
    train_dataset = LlavaDataset(processor, data_args.data_path, data_args.image_dir, data_args.ignore_index)
    collator = LlaVAForTrainCollator(data_args.max_length, data_args.ignore_index, processor.tokenizer.pad_token_id)
    # 4 trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collator,
        args=training_args
    )
    trainer.train()
    # trainer.save_state()
    trainer.save_model(output_dir=training_args.model_save_dir)


if __name__ == "__main__":
    train()
