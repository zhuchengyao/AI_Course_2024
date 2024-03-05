import sys
import numpy as np
import torch
import sentencepiece as spm


# Generate vocab
def train_model(fname, prefix):
    spm.SentencePieceTrainer.train(input=fname, model_prefix=prefix, vocab_size=16000)


corpus = "bird_shooter.txt"
prefix = "bird_shooter"
train_model(corpus, prefix)


#
def load_tokenizer(model_file):
    sp = spm.SentencePieceProcessor()
    if not sp.load(model_file=model_file):
        return False, None
    else:
        return True, sp


def load_file_into_splits(text_file, split_ratio):
    with open(text_file, 'r') as file:
        data = file.read()
    split_idx = int(len(data) * split_ratio)
    return data[:split_idx], data[split_idx:]


# encode the training and testing data
def encode_and_save(sp, content, prefix):
    token_ids = sp.encode(content, out_type=int)
    print(f"data split of {prefix} has {len(token_ids)} tokens")
    token_ids = np.array(token_ids, dtype=np.int32)
    token_ids.tofile("{}.dat".format(prefix))


def gen_dataset(text_file, model_file):
    flag, sp = load_tokenizer(model_file)
    if not flag:
        print(f"load tokenizer model from:{model_file} failed")
        sys.exit(1)
    split_ratio = 0.9
    train_text, test_text = load_file_into_splits(text_file, split_ratio)
    encode_and_save(sp, train_text, "train")
    encode_and_save(sp, test_text, "test")



def get_batch(data, batch_size=4):
    win_len = 10
    ix = torch.randint(len(data)-win_len, (batch_size,))
    x = np.stack([data[i:i+win_len] for i in ix])
    y = np.stack([data[i+1:i+1+win_len] for i in ix])
    return x, y

gen_dataset(corpus, prefix + ".model")


model_file = prefix + ".model"
def gen_samples(fname):
    train_data = np.memmap(fname, dtype=np.int32, mode='r')
    x, y = get_batch(train_data)

    flag, sp = load_tokenizer(model_file)
    if not flag:
        print(f"load tokenizer model from: {model_file} failed")
        sys.exit(1)

    for features, targets in zip(x, y):
        print("features: ", sp.decode(features.tolist()))
        print("targets : ", sp.decode(targets.tolist()))

gen_samples("train.dat")







