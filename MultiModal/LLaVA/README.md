该项目致力于如何如何训练一个属于自己的LLaVA模型，分为以下五步骤：

* 1 下载数据集和模型
  * 1.1 数据集：liuhaotian/LLaVA-Pretrain
  * 1.2 模型：
    * image: clip-vit-large-patch14-336
    * text: Qwen/Qwen1.5-7B-Chat
* 2 初始化LLaVA
  * 2.1 根据LLM Tokenizer增加<image>的token
  * 2.2 根据CLIP和Qwen替换LLaVA的image和text端
  * 2.3 替换pad_token_id和image_token_index
  * 2.4 保存LLaVA模型
  * 2.4 添加image processor, 将clip的preprocessor_config.json复制粘贴到llava文件夹下
* 3 重写Dataset和Collator
* 4 配置Trainer
* 5 deepspeed开启训练

### 1 下载数据集和模型

huggingface进行数据集和模型的下载，遇到huggingface无法访问的友友们，可以查看https://hf-mirror.com/

* clip和llm放入model文件夹下
* json数据放入data文件夹下
* image.zip解压到images文件夹下

一切配置均可进行自定义修改。

### 2 初始化LLaVA

具体可以查看"1 llava model init.ipynb"，也提供了build_model.py版本

### 3 重写Dataset和Collator

具体查看"2 buil_dataset.ipynb"

### 4 配置Trainer

具体查看llava/Trainer.py

### 5 开启训练

使用deepspeed框架进行训练，具体deepspeed的配置可以查看博客：[微调方法/量化方法与计算资源的平衡](http://124.70.193.130/微调方法-量化方法与计算资源的平衡/)

sh scripts/pretrain.sh



个人博客地址：https://crabboss.space