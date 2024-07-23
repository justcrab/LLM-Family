from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
"""
使用该脚本，将lora的权重合并大base model中
"""


def merge_lora_to_base_model():
    model_name_or_path = '/public/xcx/Item/Pre-Train/CrabGPT/outputs/pretrained/sky_5/'
    adapter_name_or_path = '/public/xcx/Item/Pre-Train/CrabGPT/model/sft/all_sft_data'
    save_path = '/public/xcx/Item/Pre-Train/CrabGPT/model/sft/total_merge_lora_model'

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map={'': 'cpu'}
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    # model = PeftModel.from_pretrained(model, adapter_name_or_path).cuda()
    model = model.merge_and_unload()
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    merge_lora_to_base_model()
