在这个快速入门中，我们将向您展示如何：

- 设置LangChain、LangSmith和LangServe
- 使用LangChain的最基本和常见的组件：提示模板、模型和输出解析器
- 使用LangChain表达语言，这是LangChain构建的协议，也是组件链接的基础
- 使用LangChain构建一个简单的应用程序
- 使用LangSmith跟踪您的应用程序
- 使用LangServe为您的应用程序提供服务

LangChain使其能够将外部数据源和计算与LLM连接起来。 在这个快速入门中，我们将介绍一些不同的方法来实现这一点。

- 1 我们从一个简单的LLM开始，依靠大模型内部的知识进行回复 **— Model I/O**
- 2 我们将从一个简单的LLM链开始，它只依赖于提示模板中的信息来回复。**— Model I/O + Chain**
- 3 接下来，我们将构建一个检索链，该链从单独的数据库获取数据并将其传递到提示模板中。 **— Model I/O + Retrievel + Chain**
- 4 然后，我们将添加聊天记录，以创建一个对话检索链。这使您可以以聊天方式与此LLM进行交互，因此它会记住以前的问题。**— Model I/O + Retrievel + Chain + Memory**
- 5 最后，我们将构建一个代理，该代理利用LLM来确定是否需要获取数据来回答问题。 我们将简要介绍这些内容，但是所有这些都有很多细节！我们会链接到相关的文档。 **— Model I/O + Retrievel + Chain + Memory + Agent**

### 1 LLM

该项目使用Deepseek API，Deepseek和OPENAI的集成很好，基本上替换一下API_KEY和BASE_URL即可无缝使用。

```
llm.invoke("how can langsmith help with testing?")
```

### 2 LLM Chain

LLM Chain = Prompt | LLM | Outpaser

单轮对话的经典实现。

### 3 LLM Retrieval Chain

* 1 Prompt | LLM
* 2 data loader | data split | embedding encode | embedding db | retriever

经典的RAG的过程，数据载入，数据切片，数据向量化，向量库，检索器。

* 1 在3的基础上增加了Memory上下文；
* 2 引入了chain的信息流动机制，此处的检索器使用了chat history和向量库的数据；

### 5 LLM Retrieval Agent

Agent最小实现。



博客地址：[langchian quickstart](http://124.70.193.130/langchian-quickstart/)