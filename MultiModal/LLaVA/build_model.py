import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaConfig

# 1 初始化Model
clip_path, llm_path = "model/clip", "/public/MountData/xcx/pretrained/model/qwen_7b"
clip_processor = AutoProcessor.from_pretrained(clip_path, device_map="cuda:0")
llm_tokenizer = AutoTokenizer.from_pretrained(llm_path, device_map="cuda:0")
clip_model = AutoModel.from_pretrained(clip_path, device_map="cuda:0")
llm_model = AutoModelForCausalLM.from_pretrained(llm_path, device_map="cuda:0", torch_dtype=torch.bfloat16)
clip_config = clip_model.vision_model.config
llm_config = llm_model.config
llava_config = LlavaConfig(vision_config=clip_config, text_config=llm_config)
llava_model = LlavaForConditionalGeneration(llava_config)

# 2 替换llava model的视觉和文本模型
llava_model.vision_tower.vision_model = clip_model.vision_model
llava_model.language_model = llm_model

# 3 替换pad_token_id和image_token_index
llava_model.config.pad_token_id = llm_tokenizer.pad_token_id
llava_model.config.image_token_index = llm_tokenizer.encode("<image>")[0]

# 4 保存模型
llava_path = "model/llava"
llava_model.save_pretrained(llava_path)
llm_tokenizer.save_pretrained(llava_path)