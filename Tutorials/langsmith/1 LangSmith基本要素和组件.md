docs：https://docs.google.com/presentation/d/1N1JKe_7qLIuiY2M40gKFGGYTb37t6F_oqGLUpw5oXy4/edit#slide=id.g26cd6bfa6fa_0_0

这里我们介绍一下LangSmith的4大基础组件：

* 1 Dataset
* 2 Evaluation
* 3 Task
* 4 Interpretion

![img](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUfJ0aafZxcnU5mlVCYbFf0nhMLdSSr76dEkNEppe4CPkGBFHacul5j_lwWEuiCSSyphYOGc-IW6OznYI9BSLn_V_0ieZZ9Y0jh3e812EmRkqLMXeBVY9JvY5l9llm8vkJQ2VCb-VboTY0Gpg1z_M1VTHNubrIZl=s2048?key=bFM9aAWTSRPt1pt49fN9YQ)

## 1 Task

LLM app: Collection of runs (trace)

LangSmith首先进入我们眼帘是因为其trace功能能够帮助我们追踪langchain中发生什么。

LangSmith将追踪过程抽象成Run, Trace和Project三项。

* Run: 最基础的底层实现，一次retriever或者一次llm回复。
* Trace：由Run组合而成的Chain的实现，例如Retrieval_chain。
* Project：由Trace组合而成的Project，多个chain组合而成的project。

![img](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUf23hCv-UEO_m-g7CZ8O24J3bMvkbYIFu4XJh6yhJHAwViYImo_IxrFlerxNv9HAkWPM9mEWT5Qlp52Ki5FJPBnNncNd9CT05Jj12vGOKI56kEtlyBLPWtFiq8ao1lbJcva5R9hrilsSvnOTH3_u7n5uyNNDpo=s2048?key=bFM9aAWTSRPt1pt49fN9YQ)

source: https://docs.smith.langchain.com/tracing/concepts

## 2 Dataset

Dataset: Collection of examples

example: Input:{"question": xxx}  Output: {"answer": yyy}

examples type:

* 1 KV pair
* 2 Chat I/O
* 3 LLM I/O

![img](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUfjJAXrzOPaiqEWmSfdG6mAtaYhFSp3iDcLnaUW2FJ7vPY8yPKGdWAWenRttYiS5qVt6PLM_6wLR6zdqw_mQ16SxacPCH38Q70bKu8Tdqz6fKeIZ_VUlun_Ptfl8Q8JDzoEoNZYemK5oPjfZnGtn7E9Y6oza4ug=s2048?key=bFM9aAWTSRPt1pt49fN9YQ)

source: https://docs.smith.langchain.com/tracing/concepts

## 3 Evaluation

Function to score LLM app on an example

Evaluation分为Judges和Modes。

* Judges：判别者，用于给生成结果进行打分和比较。
  * LLM as judge: 使用大模型作为judge
  * Heuristics: 启发式工具（执行单元，要么错要么对）
  * Human：人类判别
* Modes：评估模式，决定评估的方法
  * Comparision：比较方式，最出名的是Chat Area
  * Reference-free：不需要参考的方式，例如使用大模型判断某一句话是否包含黄赌毒信息
  * Ground Truth: 存在0/1真实标签的比较方式

![img](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUembev-9mpGllK3SejP9nzYBtKHLPGZQru_PrzJqDeCnIhZtA5zDh9MI6SGT4yYZDAr_BtQ9W8kL4iZUpx13V-ZBuIojoPaJPaDlT_bFkUXi9co61vjzB-Z5zMtb5VpLuaZ4UZsLbhf13cHr_rNFNuL1Nvrm6o=s2048?key=bFM9aAWTSRPt1pt49fN9YQ)

source: https://docs.smith.langchain.com/tracing/concepts

## 4 Information flow

这里介绍一下LangSmith的信息流动：

* 1 将Dataset的Example和Task生成的outputs一起输给Evaluator
* 2 Evaluator进行评估

**Notes：需要根据选择的Evaluator整理对应的输入。**

![img](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUcF5Ql8EeECld3MF0V0auL7R13G3S1J-ZRdv5wj0sgXQq9vvFCv88bKi-mNpeHmLES0lWXmkovzgXVEyCWvcA83EVI8gbJ_ZZG7zfXd0rolvX1SxoesIE_NCR2zLOSbt-1oVKezUkqNNyukWpEA2wESu5RyVCg=s2048?key=bFM9aAWTSRPt1pt49fN9YQ)

source: https://docs.smith.langchain.com/evaluation/faq/evaluator-implementations 