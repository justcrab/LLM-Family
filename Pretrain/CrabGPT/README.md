该项目来源于从零制作一个自己的GPT系列（GIthub众多）。

### 1 项目细节

* 1 LLM / Tokenizer
  * 该项目的LLM的模型架构为Qwen1.5(0.24b, 具体可以查看config.json)
  * 词表选择了ChatGLM3-6B（token使用np.uint16存储，节省存储）。
* 2 data
  * 预训练数据：skywork的前面20个，以及wiki百科和百度百科。
    * SkyPile-150B: https://huggingface.co/datasets/Skywork/SkyPile-150B/tree/main/data
    * [wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
    * [百度网盘](https://pan.baidu.com/s/1jIpCHnWLTNYabftavo3DVw?pwd=bwvb) 提取码: bwvb
  * 微调数据：
    * llm-wizard/alpaca-gpt4-data-zh
    * BelleGroup/train_1M_CN
    * BelleGroup/train_0.5M_CN
* 3 预处理
  * data：过滤短文本，将预训练数据处理成jsonl格式
  * model / Tokenizer
    * model:  tie_word_embedding: Embedding层和LM_head层不共享参数。
    * tokenizer: 替换ChatGLM的特殊token
* 4 dataset / collator
  * Pretrain Dataset && SFT Collator
  * SFT Dataset && SFT Collator
* 5 trainer
  * 1 warmup_steps: 2000
  * 2 weight_decay: 0.1
  * 3 learning_rate: 3e-4
  * 4 lr_sheduler_type: cosine
  * 5 min_lr_ratio: 0.1
  * 6 AdamW beta1: 0.9 beta2: 0.95
  * 7 混合精度bf16=True
  * 8 Rope base=100W

### 2 训练方式

具体训练细节可以查看博客：[从零训练一个CrabGPT](http://124.70.193.130/从零训练一个crabgpt/)。使用accelerate启动训练。

#### 2.1 预训练

sh train_scripts/pretrained/qwen3_0.4b.sh

#### 2.2 微调

LoRA微调。

sh train_scripts/sft/qwen3_0.4b.sh

### 3 大模型使用

完成预训练和微调后，有3种使用方法：

* 1 先使用merge_lora.py merge权重，再使用inference.py进行命令行推理。
* 2 先使用merge_lora.py merge权重，再使用inference_with_gradio.py进行前端推理。
* 3 在线合并lora权重，inference_with_lora.py进行推理

### 4 参考文献

- 1 https://github.com/jiahe7ay/MINI_LLM
- 2 https://github.com/charent/Phi2-mini-Chinese
- 3 https://github.com/DLLXW/baby-llama2-chinese
- 4 https://github.com/charent/ChatLM-mini-Chinese



个人博客地址：https://crabboss.space