import json
import torch
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset


class LlavaDataset(Dataset):
    def __init__(self, processor, data_path, image_dir, ignore_index):
        super().__init__()
        self.image_dir = image_dir
        self.processor = processor
        self.ignore_index = ignore_index
        self.data_list = json.load(open(data_path, "r"))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = self.data_list[item]
        conversations = data["conversations"]
        image_path = osp.join(self.image_dir, data["image"])
        human = conversations[0]["value"]
        gpt = conversations[1]["value"]
        human_input_ids = self.processor.tokenizer(human)["input_ids"]
        gpt_input_ids = self.processor.tokenizer(gpt)["input_ids"]
        input_ids = human_input_ids + gpt_input_ids
        attention_mask = [1] * len(input_ids)
        labels = [self.ignore_index] * len(human_input_ids) + gpt_input_ids
        pixel_values = self.processor.image_processor(Image.open(image_path))["pixel_values"][0]
        return (input_ids, attention_mask, labels, pixel_values)
