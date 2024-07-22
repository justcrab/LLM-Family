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

该步骤分为两步，第一步增加API的调用，第一步增加对应大模型Tokenizer文件。

* 1 找到get_completion.py，修改相应的代码

  ```
  import os
  from dotenv import load_dotenv
  
  load_dotenv()
  api_key = os.environ.get("api_key")
  base_url = os.environ.get("base_url")
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

* 2 往tokenizer文件夹中新增对应大模型的Tokenizer，因为在multichunk翻译过程中需要计算token数量。

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

#### 2.4 如何使用TranslationAgent

首先来到main.py，入口函数为translate，该函数接受两个参数，TranslationSingleInput和mode

* 1 TranslationSingleInput，当输入text的token数量大于.env设置的最大token数，会自动将TranslationSingleInput变成TranslationMultiInput

  * 1.1 TranslationSingleInput

    ```
    @dataclass
    class TranslationSingleInput:
        field: str = "AI Academic"
        country: str = "england"
        source_language: str = "chinese"
        target_language: str = "english"
        text: str = None
        translation: str = None
        reflection: str = None
        improve: str = None
    ```

  * 1.2 TranslationMultiInput(multi chunk)

    ```
    @dataclass
    class TranslationMultiInput:
        field: str = "AI Academic"
        country: str = "england"
        source_language: str = "chinese"
        target_language: str = "english"
        texts: str = None
        translations: str = None
        reflections: str = None
        improves: str = None
    ```

* `

* 2 mode，不同的mode返回的信息不一致
  * mode == "all"，将返回全部信息，包括初始翻译、翻译建议、翻译改进版本。
  * mode == "init"，只返回初始翻译
  * mode == "reflect"，只返回翻译建议
  * mode == "improve"，只返回最终翻译改进版本