tool_names = ["calculator", "amap_weather", "search"]

tools_list = [
    {
        "name": "calculator",
        "description": "a calculator to calculate math problem of two element",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "string", "description": "the first element input"},
                "b": {"type": "string", "description": "the second element input"},
                "operator": {
                    "type": "string",
                    "description": "the operator to calculate the first element and second element",
                },
            },
        },
    },
    {
        "name": "amap_weather",
        "description": "amap_weather API。获取对应城市的天气数据",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市/区具体名称，如`北京市海淀区`请描述为`海淀区`",
                }
            },
            "required": ["location"],
        },
    },
    {
        "name": "search",
        "description": "百度百科，可以用于词条查询，例如小米su7，雷军，请保证输入为词条",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "需要查询的问题"}
            },
            "required": ["question"],
        },
    },
]


def calculator(a, b, operator):
    if operator == "+":
        return int(a) + int(b)
    elif operator == "-":
        return int(a) - int(b)
    elif operator == "*":
        return int(a) * int(b)
    elif operator == "/":
        return int(a) / int(b)


def amap_weather(location):
    import pandas as pd
    import requests

    url = "https://restapi.amap.com/v3/weather/weatherInfo?city={city}&key={key}"
    city_df = pd.read_excel(
        "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/agent/AMap_adcode_citycode.xlsx"
    )

    def get_city_adcode(city_name):
        filtered_df = city_df[city_df["中文名"] == city_name]
        if len(filtered_df["adcode"].values) == 0:
            raise ValueError(
                f'location {city_name} not found, availables are {city_df["中文名"]}'
            )
        else:
            return filtered_df["adcode"].values[0]

    api_key = "your api_key"
    response = requests.get(url.format(city=get_city_adcode(location), key=api_key))
    data = response.json()
    if data["status"] == "0":
        raise RuntimeError(data)
    else:
        weather = data["lives"][0]["weather"]
        temperature = data["lives"][0]["temperature"]
        return f"{location}的天气是{weather}温度是{temperature}度。"


def search(question):
    import requests
    from lxml import etree

    url = "https://baike.baidu.com/item/" + question
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/87.0.4280.88 Safari/537.36"
    }
    rep = requests.get(url, headers=headers)
    rep.encoding = "utf-8"
    # print(rep.text)
    # print(rep.text) 输出的是网页的全部源代码
    html = etree.HTML(rep.text)
    # 通过xpath筛选出所需要的代码信息
    divs = html.xpath('//div[@class="lemmaSummary_cFhDf J-summary"]/.//text()')
    full_text = "".join(
        text.strip() for text in divs if text.strip()
    )  # 合并文本并去除多余空白
    return full_text


if __name__ == "__main__":
    # action = "calculator"
    # action_args = {"a": "12312231", "b": "873862984", "operator": "+"}
    # result = eval(action)(**action_args)
    result = search("小米su7性能配置")
    print(result)
