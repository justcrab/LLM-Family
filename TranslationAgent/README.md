该项目来源于吴恩达大佬Github开源的[translation-agent](https://github.com/andrewyng/translation-agent)，本人在其基础上增加了国产大模型API的调用功能。

### 1 环境安装

具体可以参考[translation-agent](https://github.com/andrewyng/translation-agent)，再起基础上需要安装transformers库和ZhipuAI库。

### 2 项目细节介绍

项目的整体运行细节可以查看本人博客：[TranslationAgent](http://124.70.193.130/translationagent/)

本项目将吴恩达大佬的translation-agent中写在utils.py中的逻辑进行了拆分，具体拆分逻辑如下：

* 1 llm交互：get_completion.py
* 2 输入格式：schema.py
* 3 短文本翻译：simple_translate.py
* 4 长文本翻译：complex_translate.py
* 5 文本翻译：translate.py
* 6 环境：.env

#### 2.1 如何新增大模型API调用形式

找到get_completion.py，修改相应的代码：

```
llm = os.environ.get("llm")

if llm in ["deepseek-chat"]:
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
elif llm in ["glm-4"]:
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=api_key)


def get_completion(system_message, prompt, model=llm, temperature=1.0):
    response = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
```

#### 2.2 如何设置大模型的api_key和base_url

找到.env文件，修改对应的api_key和base_url等一系列参数。

```
api_key="your api_key"
base_url="your base_url"
max_token_per_chunk=1000
llm="deepseek-chat"
```

#### 2.3 如何修改Agent的WorkFlow

找到simple_translate.py中的one_chunk_translate函数，按照顺序修改对应函数:

```
def one_chunk_translate(input: TranslationSingleInput) -> TranslationSingleInput:
    input = one_chunk_initial_translation(input)
    input = one_chunk_reflect_translation(input)
    input = one_chunk_improve_translation(input)
    return input
```

找到complex_translate.py中的multi_chunk_translate函数，按照顺序修改对应函数:

```
def multi_chunk_translate(input: TranslationMultiInput) -> TranslationMultiInput:
    input = multi_chunk_initial_translation(input)
    input = multi_chunk_reflect_translation(input)
    input = multi_chunk_improve_translation(input)
    return input
```