from tqdm import trange


def process_caption(file, test=False):
    with open(file, 'r', encoding='utf8') as f:
        lines = f.readlines()
    result = []

    for i in trange(len(lines)):
        line = lines[i]
        line = line.split()
        image_id = line[0].split('#')[0]
        if test:
            words = [word.split(':')[0] for word in line[1:]]
            caption = ''.join(words)
        else:
            caption = ''.join(line[1:])
        result.append((image_id, caption))

    return result


if __name__ == '__main__':
    # 将caption处理成统一格式
    result = []
    files = ['datasets/flickr_caption-train.txt', 'datasets/flickr_caption-val.txt', 'datasets/flickr_caption-test.txt']
    modes = [False, False, True]
    for file, mode in zip(files, modes):
        result += process_caption(file, test=mode)

    with open('datasets/flickr_caption.txt', 'w', encoding='utf8') as f:
        for image_id, caption in result:
            f.write('{}\t{}\n'.format(image_id, caption))