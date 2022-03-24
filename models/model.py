import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, BertModel, BertConfig, GPT2Config
from transformers import TextGenerationPipeline, AutoTokenizer, AutoModelWithLMHead
from transformers import AutoConfig, AutoModelWithLMHead

from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import torch.nn.functional as F
from loguru import logger


class MappingType(Enum):
    MLP = 'mlp'
    BERT = 'bert'


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class BertMapper(nn.Module):
    def __init__(self, bert_config, clip_size, prefix_size, prefix_len, constant_len):
        super(BertMapper, self).__init__()
        self.prefix_len = prefix_len
        self.prefix_size = prefix_size
        self.constant_len = constant_len
        self.bert = BertModel(config=bert_config)
        self.linear = nn.Linear(clip_size, prefix_len * prefix_size)
        self.prefix_const = nn.Parameter(torch.randn(constant_len, prefix_size), requires_grad=True)

    def forward(self, x):
        bs = x.size(0)
        # 将bs个图片向量映射成[bs, prefix_len, prefix_size]
        prefix = self.linear(x).view(-1, self.prefix_len, self.prefix_size)
        # [bs, constant_len, prefix_size]
        constant = self.prefix_const.unsqueeze(0).expand(bs, self.constant_len, self.prefix_size)
        # 将prefix向量与constant向量拼接，作为bert模型的输入
        prefix = torch.cat((prefix, constant), dim=1)
        # 输出捕获attention之后的prefix向量的输出
        out = self.bert(inputs_embeds=prefix)
        out = out.last_hidden_state[:, self.prefix_len:]
        return out


class ClipCaptionModel(nn.Module):

    def __init__(self, gpt2_path, bert_config, prefix_len=10, clip_size=512, mapping_type: MappingType = MappingType.MLP,
                 finetune_gpt2=False, constant_len=10):
        super(ClipCaptionModel, self).__init__()

        # 生成模型
        # todo 修改
        try:
            self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_path)
            logger.info('succeed to load pretrain gpt2 model')
        except:
            config = GPT2Config.from_pretrained(gpt2_path)
            self.gpt2 = GPT2LMHeadModel.from_config(config)
            logger.info('random initialize gpt2 model')
        # self.gpt2 = AutoModelWithLMHead.from_pretrained(gpt2_path)
        # 将每个图片向量[clip_size] -> [prefix_len, prefix_size]
        self.prefix_size = self.gpt2.config.n_embd
        self.prefix_len = prefix_len
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((clip_size, (self.prefix_size * prefix_len) // 2, self.prefix_size * prefix_len))
        else:
            self.clip_project = BertMapper(bert_config, clip_size, self.prefix_size, prefix_len, constant_len)
        self.finetune_gpt2 = finetune_gpt2

    def forward(self, clip_embeds, caption_ids, mask):
        """

        :param clip_embeds: 图像embedding, [bs, clip_size]
        :param caption_ids: caption的文本id, [bs, len]
        :param mask: 对于caption的文本id的attention mask, [bs, len]
        :return:
        """
        # caption_inputs_embeds:[bs, caption_len, prefix_size]
        caption_embeds = self.gpt2.transformer.wte(caption_ids)
        # prefix_embeds:[bs, prefix_len, prefix_size]
        prefix_embeds = self.clip_project(clip_embeds).view(-1, self.prefix_len, self.prefix_size)
        # embedding_cat:[bs, prefix_len+caption_len, prefix_size]
        embedding_cat = torch.cat((prefix_embeds, caption_embeds), dim=1)
        out = self.gpt2(inputs_embeds=embedding_cat, attention_mask=mask)
        # logits:[bs, prefix_len+caption_len, prefix_size]
        logits = out.logits
        return logits

    def parameters(self, recurse: bool = True):
        if self.finetune_gpt2:
            return super(ClipCaptionModel, self).parameters()
        else:
            return self.clip_project.parameters()

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        if not self.finetune_gpt2:
            self.gpt2.eval()
        return self


if __name__ == '__main__':
    config = AutoConfig.from_pretrained('../pretrain_models/gpt2/config.json')
    gpt2 = AutoModelWithLMHead.from_config(config)
    input_ids = torch.tensor([[1,2,3], [1,2,3]])
    a = gpt2(input_ids, labels=input_ids)
    print(a)