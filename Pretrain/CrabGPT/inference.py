import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM


def inference(
    model: Qwen2ForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str = "Once upon a time, ",
    max_new_tokens: int = 16
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
    response = model.generate(
        input_ids, max_new_tokens=max_new_tokens, do_sample=True, eos_token_id=tokenizer.eos_token_id,
        top_k=40, top_p=0.95, temperature=0.8, repetition_penalty=1.5
    )
    response = tokenizer.batch_decode(response)[0]
    print(response)


path = "/public/xcx/Item/Pre-Train/CrabGPT/model/sft/merge_lora_model"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
model = Qwen2ForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.float16
).to(device)
input_text = "你知道雷军吗？"
inference(model, tokenizer, input_text, max_new_tokens=256)