import numpy as np
import torch


class LlaVAForTrainCollator:
    def __init__(self, max_length, ignore_index, pad_token_id):
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        batch_max_length = [len(sampler[0]) for sampler in batch]
        final_max_length = min(max(batch_max_length), self.max_length)
        batch_input_ids, batch_attention_mask, batch_labels, batch_pixel_values = [], [], [], []
        for sampler in batch:
            # 1 input_ids, attention_mask, labels, pixel_values
            input_ids, attention_mask, labels, pixel_values = sampler
            # 2 padding
            padding_len = final_max_length - len(input_ids)
            input_ids = [self.pad_token_id] * padding_len + input_ids
            attention_mask = [0] * padding_len + attention_mask
            labels = [self.ignore_index] * padding_len + labels
            # 3 truncate
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
            # 4 batch
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            batch_pixel_values.append(pixel_values)
        # 5 tensor
        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        batch_pixel_values = torch.tensor(np.array(batch_pixel_values), dtype=torch.float)

        return {
            "input_ids": batch_input_ids, "attention_mask": batch_attention_mask,
            "labels": batch_labels, "pixel_values": batch_pixel_values
        }