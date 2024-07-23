from typing import Any, Dict, List
import torch
from transformers import DataCollatorForLanguageModeling


class PreTrainDataCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        # 找出batch中的最大长度
        lengths = [len(x) for x in batch]
        # 取出batch中的最大长度，如果超过max_seq_length，则取max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)
        # batch_max_len = self.max_seq_length

        input_ids_batch, attention_mask_batch, labels_batch = [], [], []
        for x in batch:
            x = x.tolist()
            input_ids = x
            attention_mask = [1] * len(input_ids)

            padding_len = batch_max_len - len(input_ids)
            # padding
            labels = [-100] * padding_len + input_ids
            input_ids = [self.pad_token_id] * padding_len + input_ids
            attention_mask = [0] * padding_len + attention_mask

            # truncate
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            # print(f"================")
            # print(f"input_ids len is {len(input_ids)}")
            # print(f"labels len is {len(labels)}")
            # print(f"attention_mask len is {len(attention_mask)}")
            input_ids_batch.append(input_ids)
            labels_batch.append(labels)
            attention_mask_batch.append(attention_mask)

        # 将list转换为tensor，得到最终的的模型输入
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels_batch
        }
        return inputs


class SftDataCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        # max_length = [len(x["input_ids"]) for x in batch]
        max_length = [len(x[0]) for x in batch]
        batch_max_length = min(max(max_length), self.max_seq_length)
        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        for x in batch:
            # data
            input_ids, attention_mask, target_mask = x
            # input_ids = x['input_ids']
            # target_mask = x['target_mask']
            # attention_mask = x['attention_mask']

            # padding
            padding_len = batch_max_length - len(input_ids)
            input_ids = [self.pad_token_id] * padding_len + input_ids
            attention_mask = [0] * padding_len + attention_mask
            target_mask = [0] * padding_len + target_mask

            # truncate
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            target_mask = target_mask[:self.max_seq_length]

            # append
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)
        labels_batch = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        return {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels_batch
        }