import torch
import platform
import gradio as gr
from transformers import AutoTokenizer, Qwen2ForCausalLM

if platform.system().lower() == "linux":
    path = "/public/xcx/Item/Pre-Train/CrabGPT/model/sft/total_merge_lora_model"
else:
    path = "model/sft"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
tokenizer.padding_side = "left"
model = Qwen2ForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.float16
).to(device)


def inference(
    input_text, temperature=0.8, top_p=0.95, top_k=40, num_beams=4, repetition_penalty=1.5, max_new_tokens=256
):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    response = model.generate(
        input_ids, max_new_tokens=max_new_tokens, do_sample=True, eos_token_id=tokenizer.eos_token_id,
        top_k=top_k, top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty, num_beams=num_beams
    )
    response = tokenizer.batch_decode(response)[0]
    return response


gr.Interface(
    inference,
    inputs=[
        gr.components.Textbox(lines=2, label="input_text"),
        gr.components.Slider(
                minimum=0, maximum=1, value=0.8, label="Temperature"
            ),
        gr.components.Slider(
            minimum=0, maximum=1, value=0.95, label="Top p"
        ),
        gr.components.Slider(
            minimum=0, maximum=100, step=1, value=40, label="Top k"
        ),
        gr.components.Slider(
            minimum=1, maximum=4, step=1, value=4, label="Beams"
        ),
        gr.components.Slider(
            minimum=1, maximum=2, step=0.1, value=1.5, label="Repetition_penalty"
        ),
        gr.components.Slider(
            minimum=1, maximum=2000, step=1, value=256, label="Max tokens"
        ),
    ],
    outputs=[
            gr.components.Textbox(
                lines=5,
                label="Output",
            )
        ],
    title="🔰🔰 CrabGPT-LoRA",
    description="🔰🔰 CrabGPT-LoRA"
).queue().launch(server_name="127.0.0.1", share=True)

