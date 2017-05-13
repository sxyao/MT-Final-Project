from collections import defaultdict

import numpy as np
import dynet as dy
import codecs
import nltk
from data_path import *
import argparse


class Attention:
    def __init__(self, model, training_src, training_tgt):
        self.model = model
        self.training_src, self.src_vocab, self.rsrc_vocab = self.change_word2id_genevoc(training_src)
        self.training_tgt, self.tgt_vocab, self.rtgt_vocab = self.change_word2id_genevoc_output(training_tgt)
        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)
        self.embed_size = 128
        self.src_lookup = model.add_lookup_parameters((self.src_vocab_size, self.embed_size))
        self.tgt_lookup = model.add_lookup_parameters((self.tgt_vocab_size, self.embed_size))
        self.hidden_size = 128
        self.layers = 1
        self.contextsize = self.hidden_size * 2
        self.l2r_builder = dy.GRUBuilder(self.layers, self.embed_size, self.hidden_size, model)
        self.r2l_builder = dy.GRUBuilder(self.layers, self.embed_size, self.hidden_size, model)
        self.dec_builder = dy.GRUBuilder(self.layers, self.embed_size+self.contextsize, self.hidden_size*2, model)

        self.W_y = model.add_parameters((self.tgt_vocab_size, self.hidden_size*2+self.contextsize,))
        self.b_y = model.add_parameters(self.tgt_vocab_size)

        self.attention_size = 128
        self.W1_att_e = self.model.add_parameters((self.attention_size, 2 * self.hidden_size))
        self.W1_att_f = self.model.add_parameters((self.attention_size, 2 * self.hidden_size))
        self.w2_att = self.model.add_parameters((1, self.attention_size))

        self.max_len = 50

    def __attention_mlp_batch(self, H_f_batch, h_e_batch, W1_att_e, W1_att_f, w2_att):
        # H_f_batch: (2 * hidden_size, num_step, batch_size)
        # h_e_batch: (hidden_size, batch_size)
        a_t_batch = dy.tanh(dy.colwise_add(W1_att_f * H_f_batch, W1_att_e * h_e_batch)) # (attention_size, num_step, batch_size)
        a_t_batch = w2_att * a_t_batch  # (1, num_step, batch_size)
        a_t_batch = a_t_batch[0]  # (num_step, batch_size)
        alignment_batch = dy.softmax(a_t_batch)  # (num_step, batch_size)
        c_t_batch = H_f_batch * alignment_batch  # (2 * hidden_size, batch_size)
        return c_t_batch

    def __attention_mlp(self, H_f, h_e, W1_att_e, W1_att_f, w2_att):

        # Calculate the alignment score vector
        a_t = dy.tanh(dy.colwise_add(W1_att_f * H_f, W1_att_e * h_e))
        a_t = w2_att * a_t
        a_t = a_t[0]
        alignment = dy.softmax(a_t)
        c_t = H_f * alignment
        return c_t

    # Training step over a single sentence pair
    def step_batch(self, batch):
        dy.renew_cg()

        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W1_att_e = dy.parameter(self.W1_att_e)
        W1_att_f = dy.parameter(self.W1_att_f)
        w2_att = dy.parameter(self.w2_att)

        M_s = self.src_lookup
        M_t = self.tgt_lookup
        src_sent, tgt_sent = zip(*batch)
        src_sent = zip(*src_sent)
        tgt_sent = zip(*tgt_sent)
        src_sent_rev = list(reversed(src_sent))

        # Bidirectional representations
        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()

        l2r_contexts = []
        r2l_contexts = []
        for (cw_l2r, cw_r2l) in zip(src_sent, src_sent_rev):
            l2r_state = l2r_state.add_input(dy.lookup_batch(M_s, cw_l2r))
            r2l_state = r2l_state.add_input(dy.lookup_batch(M_s, cw_r2l))
            l2r_contexts.append(l2r_state.output())  # [<S>, x_1, x_2, ..., </S>]
            r2l_contexts.append(r2l_state.output())  # [</S> x_n, x_{n-1}, ... <S>]

        # encoded_h1 = l2r_state.output()
        # tem1 = encoded_h1.npvalue()

        r2l_contexts.reverse()  # [<S>, x_1, x_2, ..., </S>]

        # Combine the left and right representations for every word
        h_fs = []
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(dy.concatenate([l2r_i, r2l_i]))

        encoded_h = h_fs[-1]

        h_fs_matrix = dy.concatenate_cols(h_fs)
        # h_fs_matrix_t = dy.transpose(h_fs_matrix)

        losses = []
        num_words = 0

        # Decoder
        c_t = dy.vecInput(self.hidden_size * 2)
        c_t.set([0 for i in xrange(self.contextsize)])
        encoded_h = dy.concatenate([encoded_h])
        dec_state = self.dec_builder.initial_state([encoded_h])
        for (cw, nw) in zip(tgt_sent[0:-1], tgt_sent[1:]):
            embed = dy.lookup_batch(M_t, cw)
            dec_state = dec_state.add_input(dy.concatenate([embed, c_t]))
            h_e = dec_state.output()
            #calculate attention
            '''
            a_t = h_fs_matrix_t * h_e
            alignment = dy.softmax(a_t)
            c_t = h_fs_matrix * alignment'''
            c_t = self.__attention_mlp_batch(h_fs_matrix, h_e, W1_att_e, W1_att_f, w2_att)
            ind_tem = dy.concatenate([h_e, c_t])
            ind_tem1 = W_y * ind_tem
            ind_tem2 = ind_tem1 + b_y
            loss = dy.pickneglogsoftmax_batch(ind_tem2, nw)  # to modify
            losses.append(loss)
            num_words += 1
        return dy.sum_batches(dy.esum(losses)), num_words

    def translate_sentence(self, sent):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W1_att_e = dy.parameter(self.W1_att_e)
        W1_att_f = dy.parameter(self.W1_att_f)
        w2_att = dy.parameter(self.w2_att)
        M_s = self.src_lookup
        M_t = self.tgt_lookup

        src_sent = sent
        src_sent_rev = list(reversed(sent))

        # Bidirectional representations
        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []
        for (cw_l2r, cw_r2l) in zip(src_sent, src_sent_rev):
            l2r_state = l2r_state.add_input(M_s[cw_l2r])
            r2l_state = r2l_state.add_input(M_s[cw_r2l])
            l2r_contexts.append(l2r_state.output())  # [<S>, x_1, x_2, ..., </S>]
            r2l_contexts.append(r2l_state.output())  # [</S> x_n, x_{n-1}, ... <S>]
        r2l_contexts.reverse()  # [<S>, x_1, x_2, ..., </S>]

        # Combine the left and right representations for every word
        h_fs = []
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(dy.concatenate([l2r_i, r2l_i]))
        encoded_h = h_fs[-1]
        h_fs_matrix = dy.concatenate_cols(h_fs)
        h_fs_matrix_t = dy.transpose(h_fs_matrix)

        # Decoder
        trans_sentence = [u'<s>']
        cw = self.tgt_vocab[u'<s>']
        c_t = dy.vecInput(self.hidden_size * 2)
        c_t.set([0 for i in xrange(self.contextsize)])
        dec_state = self.dec_builder.initial_state([encoded_h])

        while len(trans_sentence) < self.max_len:
            embed = dy.lookup(M_t,cw)
            dec_state = dec_state.add_input(dy.concatenate([embed, c_t]))
            h_e = dec_state.output()
            # c_t = self.__attention_mlp(h_fs_matrix, h_e)
            c_t = self.__attention_mlp(h_fs_matrix, h_e, W1_att_e, W1_att_f, w2_att)

            # calculate attention
            '''
            a_t = h_fs_matrix_t * h_e
            alignment = dy.softmax(a_t)
            c_t = h_fs_matrix * alignment'''
            ind_tem = dy.concatenate([h_e, c_t])
            ind_tem1 = W_y * ind_tem
            ind_tem2 = ind_tem1 + b_y
            score = dy.softmax(ind_tem2)
            probs1 = score.npvalue()
            cw = np.argmax(probs1)
            if cw == self.tgt_vocab[u'</s>']:
                break
            trans_sentence.append(self.rtgt_vocab[cw])
        return trans_sentence[1:]

    def evaluate(self, dev_src, dev_tgt, outpath):
        hypos = []
        reference = []
        fout = codecs.open(outpath, 'w', 'utf-8')
        for i in xrange(len(dev_src)):
            sen_src = dev_src[i]
            p_sen = self.translate_sentence(sen_src)
            fout.write(u' '.join(p_sen) + u'\n')
            hypos.append(p_sen)
            reference.append([dev_tgt[i]])
        chencherry = nltk.translate.bleu_score.SmoothingFunction()
        fout.close()
        BLEU = nltk.translate.bleu_score.corpus_bleu(reference, hypos, smoothing_function=chencherry.method2)
        return BLEU * 100

    def translate_corpus(self, src, outpath):
        fout = codecs.open(outpath,'w','utf-8')
        for sen_src in src:
            p_sen = self.translate_sentence(sen_src)
            fout.write(u' '.join(p_sen)+u'\n')
        fout.close()

    def change_word2id_genevoc(self,data):
        r_data = []
        vocab = defaultdict(lambda: len(vocab))
        vocab[u'<unk>'], vocab[u'<s>'], vocab[u'</s>']
        r_vocab = {0: u'<unk>', 1: u'<s>', 2: u'</s>'}
        for line in data:
            tem = []
            for word in line:
                tem.append(vocab[word])
                r_vocab[vocab[word]] = word
            r_data.append([vocab[u'<s>']]+tem+[vocab[u'</s>']])
        return [r_data,vocab,r_vocab]

    def change_word2id_genevoc_output(self,data):
        r_data = []
        counter = defaultdict(int)
        vocab = defaultdict(lambda: len(vocab))
        vocab[u'<unk>'], vocab[u'<s>'], vocab[u'</s>']
        r_vocab = {0: u'<unk>', 1: u'<s>', 2: u'</s>'}
        for line in data:
            for word in line:
                counter[word] += 1
                if counter[word]>=3:
                    vocab[word]
                    r_vocab[vocab[word]] = word
        r_data = self.change_word2id(data,vocab)
        return [r_data,vocab,r_vocab]

    def change_word2id(self, data, vocab):
        r_data = []
        for line in data:
            tem = []
            for word in line:
                if word in vocab:
                    tem.append(vocab[word])
                else:
                    tem.append(vocab[u'<unk>'])
            r_data.append([vocab[u'<s>']] + tem + [vocab[u'</s>']])
        return r_data

    # def change_id2word(self,data,r_vocab):
    #     r_data = []
    #     for line in data:
    #         tem = [r_vocab[word] for word in line]
    #         r_data.append(tem)
    #     return r_data

    def set_dropout(self, p):
        self.l2r_builder.set_dropout(p)
        self.r2l_builder.set_dropout(p)
        self.dec_builder.set_dropout(p)

    def disable_dropout(self):
        self.l2r_builder.disable_dropout()
        self.r2l_builder.disable_dropout()
        self.dec_builder.disable_dropout()


def read_file(filename, tag=None):
    data = []
    for line in codecs.open(filename, 'r', 'utf-8'):
        words = line[:-1].split(u' ')
        if tag is not None:
            words = [tag] + words
        data.append(words)
    return data


def divide_batch(data, batch_size):
    lendict = defaultdict(list)
    for item in data:
        lendict[(len(item[0]), len(item[1]))].append(item)
    n_data = []
    for k in lendict:
        value = lendict[k]
        for ind in xrange(0, len(value), batch_size):
            n_data.append(value[ind:ind + batch_size])
    return n_data


def main(argv):
    model = dy.Model()
    trainer = dy.AdamTrainer(model)
    training_src = argv[1]
    training_tgt = argv[2]
    dev_src = argv[3]
    dev_tgt = argv[4]
    test_src = argv[5]
    test_tgt = argv[6]
    dev_out = argv[7]
    test_out = argv[8]
    dev_ends = argv[9]
    test_ends = argv[10]
    batch_size = 64
    attention = Attention(model, training_src, training_tgt)
    epoch_num = 20
    train_data = zip(attention.training_src, attention.training_tgt)

    dev_src = attention.change_word2id(dev_src, attention.src_vocab)
    test_src = attention.change_word2id(test_src, attention.src_vocab)

    train_data = divide_batch(train_data, batch_size)

    for i in xrange(epoch_num):
        print 'Epoch ' + str(i + 1)
        np.random.shuffle(train_data)
        count = 1
        total = len(train_data)
        attention.set_dropout(0.5)
        for batch in train_data:
            losses, num_words = attention.step_batch(batch)
            if count % 50 == 0:
                tem = losses.value()
                print 'Iter', count, '/', total, tem / (num_words * len(batch))
            losses.backward()
            trainer.update()
            count += 1

        attention.disable_dropout()
        if dev_ends is None:
            bleuscore = attention.evaluate(dev_src, dev_tgt, 'grudp_output/' + dev_out + '_' + str(i))
            print 'Epoch', i + 1, 'Valid', bleuscore

            test_bleu = attention.evaluate(test_src, test_tgt, 'grudp_output/' + test_out + '_' + str(i))
            print 'Epoch', i + 1, 'Test', test_bleu
        else:
            start = 0
            for j, end in enumerate(dev_ends):
                bleuscore = attention.evaluate(dev_src[start: end], dev_tgt[start: end],
                                               'grudp_output/' + dev_out[j] + '_' + str(i))
                print 'Epoch', i + 1, 'Valid', bleuscore
                start = end

            start = 0
            for j, end in enumerate(test_ends):
                test_bleu = attention.evaluate(test_src[start: end], test_tgt[start: end],
                                               'grudp_output/' + test_out[j] + '_' + str(i))
                print 'Epoch', i + 1, 'Test', test_bleu
                start = end


splits = {('english', 'czech'): cs_en_split, ('czech', 'english'): cs_en_split,
          ('english', 'spanish'): es_en_split, ('spanish', 'english'): es_en_split,
          ('english', 'french'): fr_en_split, ('french', 'english'): fr_en_split}
fulls = {'cs': 'czech', 'en': 'english', 'fr': 'french', 'es': 'spanish'}


def test_single():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', type=str, default='en')
    parser.add_argument('-tgt', type=str, default='es')
    parser.add_argument('--dynet-mem', type=int, default=3000)
    parser.add_argument('--dynet-gpu-ids', type=int)
    args = parser.parse_args()
    if args.src in fulls:
        args.src = fulls[args.src]
    if args.tgt in fulls:
        args.tgt = fulls[args.tgt]

    spl = splits[(args.src, args.tgt)]
    training_src = read_file(spl[args.src]['train'])
    training_tgt = read_file(spl[args.tgt]['train'])
    dev_src = read_file(spl[args.src]['valid'])
    dev_tgt = read_file(spl[args.tgt]['valid'])
    test_src = read_file(spl[args.src]['test'])
    test_tgt = read_file(spl[args.tgt]['test'])
    dev_out = spl[args.tgt]['valid'].split('/')[-1]
    test_out = spl[args.tgt]['test'].split('/')[-1]
    argv = ['',
            training_src, training_tgt, dev_src, dev_tgt, test_src, test_tgt,
            dev_out, test_out, None, None]
    main(argv)


def test_one_to_many():
    training_src = read_file(cs_en_split['english']['train'], '<2cs>')
    training_src += read_file(es_en_split['english']['train'], '<2es>')
    training_src += read_file(fr_en_split['english']['train'], '<2fr>')
    training_tgt = read_file(cs_en_split['czech']['train'])
    training_tgt += read_file(es_en_split['spanish']['train'])
    training_tgt += read_file(fr_en_split['french']['train'])

    dev_src = read_file(cs_en_split['english']['valid'], '<2cs>')
    dev_src += read_file(es_en_split['english']['valid'], '<2es>')
    dev_src += read_file(fr_en_split['english']['valid'], '<2fr>')
    dev_ends = []
    dev_tgt = read_file(cs_en_split['czech']['valid'])
    dev_ends.append(len(dev_tgt))
    dev_tgt += read_file(es_en_split['spanish']['valid'])
    dev_ends.append(len(dev_tgt))
    dev_tgt += read_file(fr_en_split['french']['valid'])
    dev_ends.append(len(dev_tgt))

    test_src = read_file(cs_en_split['english']['test'], '<2cs>')
    test_src += read_file(es_en_split['english']['test'], '<2es>')
    test_src += read_file(fr_en_split['english']['test'], '<2fr>')
    test_ends = []
    test_tgt = read_file(cs_en_split['czech']['test'])
    test_ends.append(len(test_tgt))
    test_tgt += read_file(es_en_split['spanish']['test'])
    test_ends.append(len(test_tgt))
    test_tgt += read_file(fr_en_split['french']['test'])
    test_ends.append(len(test_tgt))

    dev_out = [cs_en_split['czech']['valid'].split('/')[-1] + '_o2m_o1',
               es_en_split['spanish']['valid'].split('/')[-1] + '_o2m_o1',
               fr_en_split['french']['valid'].split('/')[-1] + '_o2m_o1']
    test_out = [cs_en_split['czech']['test'].split('/')[-1] + '_o2m_o1',
                es_en_split['spanish']['test'].split('/')[-1] + '_o2m_o1',
                fr_en_split['french']['test'].split('/')[-1] + '_o2m_o1']
    argv = ['',
            training_src, training_tgt, dev_src, dev_tgt, test_src, test_tgt,
            dev_out, test_out, dev_ends, test_ends]
    main(argv)


if __name__ == '__main__':
    test_one_to_many()
