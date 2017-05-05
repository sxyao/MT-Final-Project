__author__ = 'yuhongliang324'
import random
from nltk.tokenize import word_tokenize
from data_path import *


def write_sents(sents, out_file, lang):
    writer = open(out_file, 'w')
    print lang
    for i, sent in enumerate(sents):
        if (i + 1) % 10000 == 0:
            print i + 1
        sent = sent.decode('utf-8')
        toks = word_tokenize(sent, language=lang)
        tokenized_sent = ' '.join(toks)
        tokenized_sent = tokenized_sent.lower()
        tokenized_sent = tokenized_sent.encode('utf-8')
        writer.write(tokenized_sent + '\n')
    writer.close()


def sample(origin, split, src_lang, tgt_lang, n_train=100000, n_val=1000, n_test=1000):

    reader = open(origin[src_lang])
    src_sents_old = reader.readlines()
    reader.close()
    src_sents_old = map(lambda x: x.strip(), src_sents_old)

    reader = open(origin[tgt_lang])
    tgt_sents_old = reader.readlines()
    reader.close()
    tgt_sents_old = map(lambda x: x.strip(), tgt_sents_old)

    src_sents, tgt_sents = [], []

    for src_sent, tgt_sent in zip(src_sents_old, tgt_sents_old):
        if len(src_sent) < 5 or len(tgt_sent) < 5:
            continue
        if src_sent.startswith('<') or tgt_sent.startswith('<'):
            continue
        src_sents.append(src_sent)
        tgt_sents.append(tgt_sent)

    print len(src_sents), len(tgt_sents)

    sents = zip(src_sents, tgt_sents)
    random.shuffle(sents)

    src_sents_train = [sent[0] for sent in sents[:n_train]]
    tgt_sents_train = [sent[1] for sent in sents[:n_train]]

    src_sents_valid = [sent[0] for sent in sents[n_train: n_train + n_val]]
    tgt_sents_valid = [sent[1] for sent in sents[n_train: n_train + n_val]]

    src_sents_test = [sent[0] for sent in sents[n_train + n_val: n_train + n_val + n_test]]
    tgt_sents_test = [sent[1] for sent in sents[n_train + n_val: n_train + n_val + n_test]]

    write_sents(src_sents_train, split[src_lang]['train'], src_lang)
    write_sents(tgt_sents_train, split[tgt_lang]['train'], tgt_lang)

    write_sents(src_sents_valid, split[src_lang]['valid'], src_lang)
    write_sents(tgt_sents_valid, split[tgt_lang]['valid'], tgt_lang)

    write_sents(src_sents_test, split[src_lang]['test'], src_lang)
    write_sents(tgt_sents_test, split[tgt_lang]['test'], tgt_lang)


def sample_non_english(origin_src, origin_tgt, split, src_lang, tgt_lang, n_train=100000, n_val=1000, n_test=1000):
    def get_correspond(origin, src_lang):
        reader = open(origin[src_lang])
        src_sents = reader.readlines()
        reader.close()
        src_sents = map(lambda x: x.strip(), src_sents)

        reader = open(origin['english'])
        eng_sents = reader.readlines()
        reader.close()
        eng_sents = map(lambda x: x.strip(), eng_sents)

        eng_frg = {}
        for eng, frg in zip(eng_sents, src_sents):
            eng_frg[eng] = frg
        return eng_frg

    eng_src = get_correspond(origin_src, src_lang)
    eng_tgt = get_correspond(origin_tgt, tgt_lang)
    sents = []
    for eng, src in eng_src.items():
        if eng in eng_tgt:
            sents.append((src, eng_tgt[eng]))

    random.shuffle(sents)

    src_sents_train = [sent[0] for sent in sents[:n_train]]
    tgt_sents_train = [sent[1] for sent in sents[:n_train]]

    src_sents_valid = [sent[0] for sent in sents[n_train: n_train + n_val]]
    tgt_sents_valid = [sent[1] for sent in sents[n_train: n_train + n_val]]

    src_sents_test = [sent[0] for sent in sents[n_train + n_val: n_train + n_val + n_test]]
    tgt_sents_test = [sent[1] for sent in sents[n_train + n_val: n_train + n_val + n_test]]

    write_sents(src_sents_train, split[src_lang]['train'], src_lang)
    write_sents(tgt_sents_train, split[tgt_lang]['train'], tgt_lang)

    write_sents(src_sents_valid, split[src_lang]['valid'], src_lang)
    write_sents(tgt_sents_valid, split[tgt_lang]['valid'], tgt_lang)

    write_sents(src_sents_test, split[src_lang]['test'], src_lang)
    write_sents(tgt_sents_test, split[tgt_lang]['test'], tgt_lang)


def test1():
    # sample(cs_en_origin, cs_en_split, 'czech', 'english')
    # sample(es_en_origin, es_en_split, 'spanish', 'english')
    sample(fr_en_origin, fr_en_split, 'french', 'english')


if __name__ == '__main__':
    test1()
