import json


with open("/public/MountData/xcx/pretrained/wiki_baidu/baidubaike.jsonl", "r") as f:
    data = f.readlines()

data1 = data[:len(data)//3]
data2 = data[len(data)//3:len(data)*2//3]
data3 = data[len(data)*2//3:]

with open("/public/MountData/xcx/pretrained/wiki_baidu/baidubaike_0.jsonl", "w") as f:
    f.writelines(data1)

with open("/public/MountData/xcx/pretrained/wiki_baidu/baidubaike_1.jsonl", "w") as f:
    f.writelines(data2)

with open("/public/MountData/xcx/pretrained/wiki_baidu/baidubaike_2.jsonl", "w") as f:
    f.writelines(data3)



