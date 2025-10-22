# from huggingface_hub import hf_hub_download
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AutoModel, AutoTokenizer



# Tải Tokenizer và Model
phobert = AutoModel.from_pretrained("vinai/PhoBERT-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/PhoBERT-base-v2")

# Ví dụ về câu đã được tách từ
sentence = 'Chúng_tôi là những sinh_viên .'

# Tokenize và mã hóa
input_ids = torch.tensor([tokenizer.encode(sentence)])

with torch.no_grad():
    features = phobert(input_ids)
    # Lấy hidden states của lớp cuối cùng
    last_hidden_states = features.last_hidden_state
    
    
# Lấy hidden state của token [CLS] (ở vị trí 0)
cls_embedding = last_hidden_states[:, 0, :]

# cls_embedding sẽ có kích thước (1, 768) với mô hình base, là vector embedding của câu.
print("PhoBERT Embedding:", cls_embedding)
print("Embedding shape:", cls_embedding.shape)