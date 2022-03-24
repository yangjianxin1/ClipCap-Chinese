import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, BertTokenizer
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from os.path import join
from loguru import logger
import glob
import skimage.io as io
from PIL import Image


class ClipCapDataset(Dataset):
    def __init__(self, clip_data_path, prefix_len, tokenizer, max_len, mode='train', normalize_prefix=False):
        assert mode in ['train', 'test']
        self.normalize_prefix = normalize_prefix
        pad_id = tokenizer.pad_token_id
        if mode == 'train':
            save_path = join(os.path.dirname(clip_data_path), 'train.pkl')
        else:
            save_path = join(os.path.dirname(clip_data_path), 'test.pkl')

        # 加载缓存
        if os.path.isfile(save_path):
            with open(save_path, 'rb') as f:
                self.clip_embeds, self.caption_ids_list, self.mask_list = pickle.load(f)
            logger.info('num of training data'.format(len(self.clip_embeds)))
        else:
            logger.info('loading dataset:{}'.format(clip_data_path))
            with open(clip_data_path, 'rb') as f:
                caption_list, image_id2embed = pickle.load(f)
            logger.info('num of image embedding:{}'.format(len(image_id2embed)))
            logger.info('num of captions:{}'.format(len(caption_list)))

            clip_embeds = []
            caption_ids_list = []
            mask_list = []
            for image_id, caption in caption_list:
                clip_embed = image_id2embed[image_id].squeeze(0).float()
                caption_ids = tokenizer.encode(caption, add_special_tokens=False)
                caption_ids.append(tokenizer.sep_token_id)

                # truncate
                caption_ids = caption_ids[:max_len-prefix_len]
                mask = [1] * (prefix_len + len(caption_ids))

                # padding
                padding_len = max_len - prefix_len - len(caption_ids)
                caption_ids += [pad_id]*padding_len
                mask += [0]*padding_len

                caption_ids = torch.tensor(caption_ids).long()
                mask = torch.tensor(mask).long()

                clip_embeds.append(clip_embed)
                caption_ids_list.append(caption_ids)
                mask_list.append(mask)
            with open(save_path, 'wb') as f:
                pickle.dump([clip_embeds, caption_ids_list, mask_list], f)
            self.clip_embeds = clip_embeds
            self.caption_ids_list = caption_ids_list
            self.mask_list = mask_list
            logger.info('num of training data'.format(len(self.clip_embeds)))

    def __len__(self) -> int:
        return len(self.caption_ids_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        clip_embed = self.clip_embeds[index]
        caption_ids = self.caption_ids_list[index]
        mask = self.mask_list[index]
        if self.normalize_prefix:
            clip_embed = clip_embed / clip_embed.norm(2, -1)    # todo check
        return clip_embed, caption_ids, mask


class ImageDataset(Dataset):
    def __init__(self, path, preprocess):
        # 加载路径下的所有图片
        self.images = []
        self.image_names = []
        for file in glob.glob(join(path, '*')):
            image = io.imread(file)
            image = preprocess(Image.fromarray(image)).squeeze(0)
            filename = os.path.basename(file)
            self.images.append(image)
            self.image_names.append(filename)

    def __getitem__(self, item):
        return self.images[item], self.image_names[item]

    def __len__(self) -> int:
        return len(self.images)
