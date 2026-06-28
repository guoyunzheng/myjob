import torch
from torch import nn
import transformers
import os
CLIP_PATH = "./clip1"
if not os.path.exists(CLIP_PATH):
    print(f"⚠️  警告: CLIP模型路径 {CLIP_PATH} 不存在")
class ClipTokenizer:

    def __init__(self):
        super().__init__()
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
            CLIP_PATH,  # 使用本地路径
            local_files_only=True  # 强制只使用本地文件
        )

    @torch.inference_mode()
    def __call__(self, instructions):
        return self.tokenizer(
            instructions,
            padding="longest",
            return_tensors="pt"
        )["input_ids"]


class ClipTextEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = transformers.CLIPTextModel.from_pretrained(
            CLIP_PATH,  # 使用本地路径
            local_files_only=True  # 强制只使用本地文件
        )

    def forward(self, tokens):
        return self.model(tokens).last_hidden_state
