import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import argparse
from tqdm import tqdm, trange
from os.path import join
from loguru import logger


def main(args):
    device = torch.device('cuda:0')
    clip_model, preprocess = clip.load(args.clip_model_path, device=device, jit=False)
    with open(args.caption_path, 'r') as f:
        lines = f.readlines()

    logger.info('len of captions：{}'.format(len(lines)))
    image_id2embed = {}    # imageid到image embedding的映射
    caption_list = []

    for i in trange(len(lines)):
        line = lines[i].strip()
        image_id, caption = line.split('\t')

        if image_id not in image_id2embed.keys():
            # 加载caption对应的图片
            file = join(args.image_path, '{}.jpg'.format(image_id))
            # 将图片编码成embedding
            image = io.imread(file)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                clip_embed = clip_model.encode_image(image).cpu()
            image_id2embed[image_id] = clip_embed
        caption_list.append((image_id, caption))

    with open(args.output_path, 'wb') as f:
        pickle.dump([caption_list, image_id2embed], f)

    logger.info('num of image embedding:{}'.format(len(image_id2embed)))
    logger.info('num of captions:{}'.format(len(caption_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_path', default="pretrain_models/ViT-B-32.pt")
    parser.add_argument('--caption_path', default="datasets/flickr_caption.txt")
    parser.add_argument('--image_path', default="datasets/flickr30k-images")
    parser.add_argument('--output_path', default="datasets/clip_caption.pkl")
    args = parser.parse_args()
    main(args)



