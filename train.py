import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertTokenizer
from tqdm import tqdm
import os
import argparse
from dataset import ClipCapDataset
from models.model import ClipCaptionModel
import time
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from os.path import join
import transformers
import torch.nn.functional as F


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='datasets/clip_caption.pkl')
    parser.add_argument('--gpt2_path', default='pretrain_models/gpt2')
    parser.add_argument('--bert_path', default='pretrain_models/bert')
    parser.add_argument('--output_path', default='output')
    parser.add_argument("--lr", type=float, default=2e-5, help='学习率')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--prefix_len', type=int, default=10)
    parser.add_argument('--constant_len', type=int, default=10)
    parser.add_argument('--clip_size', type=int, default=512)
    parser.add_argument('--bs_train', type=int, default=2)
    parser.add_argument('--dev_size', type=int, default=10)
    parser.add_argument('--bs_eval', type=int, default=128)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument("--save_step", type=int, default=1000, help="训练多少步，保存一次模型")
    parser.add_argument("--eval_step", type=int, default=100, help="训练多少步,记录一次指标")
    parser.add_argument('--finetune_gpt2', help='finetune gpt2', action='store_true', default=False)
    parser.add_argument('--mapping_type', type=str, default='mlp', choices=['mlp', 'bert'], help='mlp or bert')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument("--do_train", action='store_true', default=True)
    # parser.add_argument("--do_test", action='store_true', default=True)
    args = parser.parse_args()
    return args


def train(model, train_loader, dev_dataloader, optimizer, scheduler, args):
    model.train()
    logger.info("start training")
    device = args.device
    for epoch in range(args.epochs):
        logger.info('start {}-th epoch training'.format(epoch + 1))
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx + 1
            clip_embeds, caption_ids, mask = data
            clip_embeds = clip_embeds.to(device).float()
            caption_ids = caption_ids.to(device)
            mask = mask.to(device)
            logits = model(clip_embeds, caption_ids, mask)

            # 计算loss
            shift_logits = logits[..., args.prefix_len - 1:-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = caption_ids.view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % args.eval_step == 0:
                dev_loss = evaluate(args, model, dev_dataloader)
                writer.add_scalar('loss', dev_loss, step)
                logger.info('loss at step {} is {}'.format(step, dev_loss.item()))
                model.train()

            if step % args.save_step == 0:
                logger.info('saving checkpoint at step {}'.format(step))
                save_path = join(args.output_path, 'checkpoint-{}.pt'.format(step))
                torch.save(model.state_dict(), save_path)


def evaluate(args, model, dataloader):
    """
    计算数据集上的指标
    :return:
    """
    model.eval()
    device = args.device
    logger.info("Running evaluation")
    eval_loss = 0.0  #
    with torch.no_grad():
        for data in tqdm(dataloader):
            clip_embeds, caption_ids, mask = data
            clip_embeds = clip_embeds.to(device).float()
            caption_ids = caption_ids.to(device)
            mask = mask.to(device)
            logits = model(clip_embeds, caption_ids, mask)

            # 计算loss
            shift_logits = logits[..., args.prefix_len - 1:-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = caption_ids.view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels)

            loss = loss.mean()  # 对多卡的loss取平均
            eval_loss += loss
    return eval_loss


def main(args):
    # 分词器
    tokenizer = BertTokenizer.from_pretrained(args.gpt2_path)
    # BertMapper的模型配置
    bert_config = BertConfig.from_pretrained(args.bert_path)
    # 加载模型
    model = ClipCaptionModel(
        args.gpt2_path, bert_config, args.prefix_len, args.clip_size, args.mapping_type,
        args.finetune_gpt2, args.constant_len
    ).to(args.device)

    if args.do_train:
        # 加载数据集
        dataset = ClipCapDataset(args.data_path, args.prefix_len, tokenizer, args.max_len, 'train', args.normalize_prefix)
        train_dataset, dev_dataset = torch.utils.data.random_split(dataset, [len(dataset) - args.dev_size, args.dev_size])
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs_train, shuffle=True, num_workers=args.num_workers)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.bs_eval, shuffle=True, num_workers=args.num_workers)
        t_total = len(train_dataloader) * args.epochs
        optimizer = transformers.AdamW(model.parameters(), lr=args.lr)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        train(model, train_dataloader, dev_dataloader, optimizer, scheduler, args)
    # if args.do_test:
    #     # 加载数据集
    #     test_dataset = ClipCapDataset(args.data_path, args.prefix_len, tokenizer, args.max_len, 'test',
    #                                    args.normalize_prefix)
    #     test_dataloader = DataLoader(test_dataset, batch_size=args.bs_test, shuffle=False,
    #                                  num_workers=args.num_workers)


if __name__ == '__main__':
    args = set_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.do_train:
        cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
        logger.info(args)
        writer = SummaryWriter(args.output_path)
    main(args)
