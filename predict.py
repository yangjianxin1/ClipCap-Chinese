import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer
from tqdm import tqdm
import os
import argparse
from dataset import ImageDataset
from models.model import ClipCaptionModel
from loguru import logger
from os.path import join
import torch.nn.functional as F
import clip


def topk_filtering(logits, topk=10, topp=0, filter_value=-float('Inf')):
    # todo topp
    """
    将topk以外的token的生成概率置为-inf
    :param logits: [b_size, dim]
    :param topk:
    :param filter_value:
    :return:
    """
    assert logits.dim() == 2  # batch size 1 for now - could be updated for more but the code would be less clear
    topk = min(topk, logits.size(-1))  # Safety check
    if topk > 0:
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        indices_to_remove = logits < torch.topk(logits, topk, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if topp > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > topp
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # todo check
        for i in range(sorted_indices_to_remove.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def generate(model, clip_embeds, tokenizer, args):
    """

    :param model:
    :param clip_embeds: [b_size x clip_size]
    :param max_len:
    :param device:
    :return:
    """
    b_size = clip_embeds.size(0)
    device = args.device
    pad_id = tokenizer.pad_token_id
    sep_id = tokenizer.sep_token_id
    unk_id = tokenizer.unk_token_id
    max_len = args.max_len
    temperature = args.temperature
    topk = args.topk
    topp = args.topp

    cur_len = 0
    caption_ids = []    # 存储生成的caption

    # gpt2模型的输入: inputs_embeds:[bs, prefix_len, prefix_size]
    inputs_embeds = model.clip_project(clip_embeds).view(-1, model.prefix_len, model.prefix_size)
    finish_flag = [False] * b_size  # 第i个输入是否完成生成的标志

    while True:
        out = model.gpt2(inputs_embeds=inputs_embeds)
        logits = out.logits  # [b_size, len, vocab_size]
        next_token_logits = logits[:, -1, :]    # 取最后一个单词的预测分布
        next_token_logits = next_token_logits / temperature
        next_token_logits[:, unk_id] = -float('Inf')   # 将unk设为无穷小

        # topk filter
        filtered_logits = topk_filtering(next_token_logits, topk, topp)
        next_token_ids = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(1).tolist()

        # 分别判断生成图片是否已生成完毕
        for index in range(len(next_token_ids)):
            token_id = next_token_ids[index]
            # 如果第i个句子已经生成结束
            if finish_flag[index]:
                next_token_ids[index] = pad_id
            # 如果第i个句子生成结束
            elif token_id == sep_id:
                finish_flag[index] = True
            # 未结束生成
            elif cur_len == 0:
                caption_ids.append([token_id])
            else:
                caption_ids[index].append(token_id)
        next_token_ids = torch.tensor(next_token_ids).to(device)
        next_token_embeds = model.gpt2.transformer.wte(next_token_ids).to(device).unsqueeze(1)
        inputs_embeds = torch.cat((inputs_embeds, next_token_embeds), dim=1)

        cur_len += 1
        if cur_len > max_len or False not in finish_flag:
            break

    # 对token_id进行解码
    captions = []
    for caption_id in caption_ids:
        caption = tokenizer.convert_ids_to_tokens(caption_id)
        caption = ''.join(caption)
        captions.append(caption)

    return captions


def main(args):
    # 分词器
    tokenizer = BertTokenizer.from_pretrained(args.gpt2_model_path)
    # BertMapper的模型配置
    bert_config = BertConfig.from_pretrained(args.bert_model_path)
    # 初始化模型
    model = ClipCaptionModel(
        args.gpt2_model_path, bert_config, args.prefix_len, args.clip_size, args.mapping_type,
        args.finetune_gpt2, args.constant_len
    ).to(args.device)
    # 加载权重
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    # 加载clip模型
    clip_model, preprocess = clip.load(args.clip_model_path, device=args.device, jit=False)

    # 加载数据集
    dataset = ImageDataset(args.image_path, preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    logger.info('start predicting')

    captions_generate = []
    image_name_list = []
    for batch_idx, data in enumerate(tqdm(dataloader)):
        images, image_names = data
        clip_embeds = clip_model.encode_image(images)
        clip_embeds = clip_embeds.unsqueeze(1).repeat(1, args.num_generate, 1).view(-1, clip_embeds.size(-1))
        captions = generate(model, clip_embeds, tokenizer, args)

        # 每num_generate个caption对应一张图片
        captions = ['\t'.join(captions[i: i+args.num_generate]) for i in range(0, clip_embeds.size(0), args.num_generate)]
        captions_generate += captions
        image_name_list += image_names

    if args.finetune_gpt2:
        save_path = join(args.output_path, 'caption_generate_finetune.txt')
    else:
        save_path = join(args.output_path, 'caption_generate_no_finetune.txt')
    with open(save_path, 'w', encoding='utf8') as f:
        for caption, image_name in zip(captions_generate, image_name_list):
            f.write('{}\t{}\n'.format(image_name, caption))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='datasets/test')
    parser.add_argument('--model_path', default='output/mlp_finetune/checkpoint-35000.pt')
    parser.add_argument('--gpt2_model_path', default="pretrain_models/gpt2")
    parser.add_argument('--bert_model_path', default="pretrain_models/bert")
    parser.add_argument('--clip_model_path', default="pretrain_models/ViT-B-32.pt")
    parser.add_argument('--output_path', default='output')
    parser.add_argument('--prefix_len', type=int, default=10)
    parser.add_argument('--constant_len', type=int, default=10)
    parser.add_argument('--clip_size', type=int, default=512)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=0, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0.8, type=float, required=False, help='')
    parser.add_argument('--num_generate', default=2, type=int, required=False, help='对于每张图片，生成多少个候选caption')
    parser.add_argument('--finetune_gpt2', help='finetune gpt2', action='store_true', default=False)
    parser.add_argument('--mapping_type', type=str, default='mlp', choices=['mlp', 'bert'], help='mlp or bert')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    main(args)
