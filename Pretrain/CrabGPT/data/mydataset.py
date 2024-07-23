import time
import json
import torch
import os.path
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from multiprocessing import Pool


class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        return data


class PreTrainDataset():
    def __init__(self, tokenizer, data_path, data_batch,
                 sliding_window, min_seq_length, cache_path, eval_size=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.data_batch = data_batch
        self.sliding_window = sliding_window
        self.min_seq_length = min_seq_length
        self.cache_path = cache_path
        self.eval_size = eval_size
        self.num_processes = 32

    def load_files_from_path(self):
        files = []
        for root, dir_names, file_names in os.walk(self.data_path):
            print(f"{root} {dir_names} file_names is {file_names}")
            for file_name in file_names:
                file = os.path.join(root, file_name)
                if file.endswith(".jsonl"):
                    files.append(file)
                else:
                    print(f"load unseen file type (not .jsonl), please check the file type")
        return files

    def load_texts_from_file(self, file) -> List[str]:
        texts_list = []
        if file.endswith(".jsonl"):
            with open(file, "r", encoding="utf-8") as f:
                data_list = f.readlines()
                for data in data_list:
                    json_data = json.loads(data)["text"] + "</s>"
                    texts_list.append(json_data)
        return texts_list

    def load_from_disk(self, file):
        try:
            if os.path.exists(file):
                try:
                    obj = pd.read_parquet(file, engine='pyarrow').text.tolist()
                    return obj
                except Exception as e:
                    print(f"pickle load failed and details is {e}")
        except Exception as e:
            print(f"file is not exists and details is {e}")

    def save_to_disk(self, obj, file):
        try:
            data = pd.DataFrame({"text": obj})
            data.to_parquet(file, engine='pyarrow', index=False)
        except Exception as e:
            print(f"pickle dump failed and details is {e}")

    def truncate_windows(self, input_ids):
        windows = []
        for i in range(0, len(input_ids), self.sliding_window):
            x = input_ids[i: i+self.sliding_window]
            if len(x) < self.min_seq_length:
                continue
            windows.append(x)
        return windows

    def tokenize_encode(self, texts):
        return self.tokenizer(texts).input_ids

    def parallel_process(self, input_list, num_processes=None):
        new_result_list = []
        with Pool(processes=num_processes) as pool:
            result_list = pool.map(self.tokenize_encode, input_list)
        for result in result_list:
            new_result_list.append(np.array(result, dtype=np.uint16))
        return new_result_list

    def load_dataset(self):
        """
        file -> json_batch -> truncate_slide_windows
        if there are 3 files: 1.jsonl(10 lines, 10 windows per lines) 2.jsonl(20 lines) 3.jsonl(30 lines)
        file : ["", "", ""] length is 3
        json_batch : [""*10, ""*20, ""*30] length is 60
        truncate_slide_windows: [""*100, ""*200, ""*300] length is 600
        """
        if os.path.exists(self.cache_path):
            data = self.load_from_disk(self.cache_path)
            dataset = MyDataset(data)
        else:
            # 1 find all file in data/pretrain
            files = self.load_files_from_path()
            print(f"total pretrain file nums is {len(files)}")

            # 2 load texts from file -> ["file1 content"*(file1 josnl length), "file2 content"*(file2 josnl length)]
            print(f"start load all pretrain file")
            texts_list = []
            for file in tqdm(files):
                texts = self.load_texts_from_file(file)
                texts_list += texts
            print(f"total pretrain data nums is {len(texts_list)}")

            # 3 tokenize texts and truncate
            print(f"start tokenize all pretrain data")
            windows_truncate_texts_list = []
            for i in tqdm(range(0, len(texts_list), self.data_batch)):
                text_list = texts_list[i: i+self.data_batch]
                for x in text_list:
                    windows_truncate_texts = self.truncate_windows(x)
                    windows_truncate_texts_list += windows_truncate_texts
            print(f"total pretrian data slide windows num is {len(windows_truncate_texts_list)}")
            del texts_list

            # 4 multiprocess to tokenizer
            start_time = time.time()
            windows_truncate_texts_list = self.parallel_process(windows_truncate_texts_list, self.num_processes)
            end_time = time.time()
            print(f"multiprocess tokenizer used time is {end_time - start_time} s")

            # 5 save windows_truncate_texts_list to disk
            self.save_to_disk(windows_truncate_texts_list, self.cache_path)

            # 6 dataset
            dataset = MyDataset(windows_truncate_texts_list)

        print('Spliting train and eval dataset ...')
        if self.eval_size == 0:
            train_dataset = dataset
            eval_dataset = None
            print(f'Num of train data: {len(train_dataset)}')
            print(f'Num of eval data: 0')
        else:
            train_dataset, eval_dataset = train_test_split(dataset, test_size=self.eval_size)
            print(f'Num of train data: {len(train_dataset)}')
            print(f'Num of eval data: {len(eval_dataset)}')

        total_token_num = 0
        for x in tqdm(train_dataset):
            total_token_num += len(x)
        print(f'Total training token num: {total_token_num}')
        return train_dataset, eval_dataset


template = {
    "template_name": "CrabGPT",
    "system_format": "{content}",
    "user_format": "<human>: {content} </human>\n<bot>: ",
    "assistant_format": "{content} </bot></s>",
    "system": "You are CrabGPT, a large language model trained by Crabboss from Zjut. Follow the user's instructions carefully.",
    "stop_word": "</s>",
}


class SftTrainDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_len):
        super(SftTrainDataset, self).__init__()
        self.file = file
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.template_name = template["template_name"]
        self.system_format = template["system_format"]
        self.user_format = template["user_format"]
        self.assistant_format = template["assistant_format"]
        self.system = template["system"]

        with open(file, "r", encoding="utf-8") as f:
            self.data_list = json.load(f)

    def __getitem__(self, index):
        data = self.data_list[index]
        if self.system:
            system_text = self.system_format.format(content=self.system)
            system_input_ids = self.tokenizer.encode(system_text)
            system_target_mask = [0] * len(system_input_ids)
        else:
            system_input_ids, system_target_mask = [], []

        instruction, input, output = data["instruction"], data["input"], data["output"]
        user_text = instruction + input
        assistant_text = output
        user_input_ids = self.tokenizer.encode(user_text)
        user_target_mask = [0] * len(user_input_ids)
        assistant_input_ids = self.tokenizer.encode(assistant_text)
        assistant_target_mask = [1] * len(assistant_input_ids)

        input_ids = system_input_ids + user_input_ids + assistant_input_ids
        attention_mask = [1] * len(input_ids)
        target_mask = system_target_mask + user_target_mask + assistant_target_mask

        input_ids = input_ids[: self.max_seq_len]
        attention_mask = attention_mask[: self.max_seq_len]
        target_mask = target_mask[: self.max_seq_len]

        assert len(input_ids) == len(attention_mask) == len(target_mask)
        # inputs = {
        #     "input_ids": input_ids,
        #     "target_mask": target_mask,
        #     "attention_mask": attention_mask
        # }
        return (input_ids, attention_mask, target_mask)

    def __len__(self):
        return len(self.data_list)
