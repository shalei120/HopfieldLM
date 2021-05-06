# Copyright 2020 . All Rights Reserved.
# Author : Lei Sha
import functools
print = functools.partial(print, flush=True)
from TranslationModel import TranslationModel

from textdata_MT import  TextData_MT
import time, sys,datetime
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import  tqdm
import os
import math,random
import nltk
from nltk.corpus import stopwords
import argparse
# import GPUtil
# import turibolt as bolt
import pickle
from Hyperparameters import args
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# import matplotlib.pyplot as plt
import numpy as np
import copy
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# from kenLM import LMEvaluator as LMEr

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--batch', '-b')
parser.add_argument('--modelarch', '-m')
parser.add_argument('--data', '-d')
parser.add_argument('--server', '-s')
parser.add_argument('--embeddingsize', '-emb')
parser.add_argument('--layernum', '-layer')
parser.add_argument('--nhead', '-nhead')

cmdargs = parser.parse_args()

usegpu = True

if cmdargs.gpu is None:
    usegpu = False
    args['device'] = 'cpu'
else:
    usegpu = True
    args['device'] = 'cuda:' + str(cmdargs.gpu)

if cmdargs.batch is None:
    pass
else:
    args['batchSize'] = int(cmdargs.batch)

if cmdargs.modelarch is None:
    pass
else:
    args['LMtype'] = cmdargs.modelarch

if cmdargs.data is None:
    pass
else:
    args['corpus'] = cmdargs.data

if cmdargs.server is None:
    args['server'] = 'other'
else:
    args['server'] = cmdargs.server

if cmdargs.embeddingsize is None:
    pass
else:
    args['embeddingSize'] = int(cmdargs.embeddingsize)


if cmdargs.layernum is None:
    pass
else:
    args['numLayers'] = int(cmdargs.layernum)
if cmdargs.nhead is None:
    args['nhead'] = 1
else:
    args['nhead'] = int(cmdargs.nhead)
args['typename'] = args['corpus']

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), datetime.datetime.now())

class Runner:
    def __init__(self):
        self.model_path = args['rootDir'] + '/model.pth'
        # self.model_bw_path = args['rootDir'] + '/model_bw.pth'
        self.drawcount = 0



    def main(self):
        self.textData =  TextData_MT('MT')

        self.start_token = self.textData.word2index[args['typename']]['START_TOKEN']
        self.end_token = self.textData.word2index[args['typename']]['END_TOKEN']
        args['vocabularySize'] = self.textData.getVocabularySize(args['typename'])
        print(self.textData.getVocabularySize(args['typename']))
        print(args)
        torch.manual_seed(0)
        self.model = TranslationModel(self.textData.word2index[args['typename']], self.textData.index2word[args['typename']]).to(args['device'])
        # self.model = torch.load(self.model_path.replace('model', 'model_'+'fw'), map_location=args['device'])
        params = sum([np.prod(p.size()) for p in self.model.parameters()])
        print(params, sum([sys.getsizeof(p.storage()) for p in self.model.parameters()])/1000000, 'm')
        self.trainMT()     # contain  model saving
        self.evaluateRandomly()
        self.testMT()

     

    def trainMT(self, print_every=10000, plot_every=10, learning_rate=0.001):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        print(type(self.textData.word2index[args['typename']]), args['device'])

        optimizer = optim.Adam(self.model.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=5.0)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        iter = 1
        batches = self.textData.getBatches(args['typename'])
        n_iters = len(batches)
        print('niters ',n_iters)

        args['trainseq2seq'] = False

        min_BLEU = -1

        # self.Cal_BLEU_for_dataset('test')

        CE_loss_total = 0
        KL_total = 0
        VAE_recon_total = 0
        error_total = 0
        print('begin training...')
        for epoch in range(args['numEpochs']):
            losses = []

            for batch in batches:
                optimizer.zero_grad()
                x={}
                x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs))
                x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs))
                x['dec_len'] = batch.decoder_lens
                x['dec_target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs))


                # print(x['enc_input'][0],x['dec_input'][0],x['dec_target'][0])
                if args['LMtype'] == 'energy':
                    data = self.model(x)    # batch seq_len outsize
                    loss_mean = torch.mean(data['loss']) + 0.1 * data['KL'] + 0.01 * data['VAE_recon'] + 1 * data['error']
                    CE_loss_total += torch.mean(data['loss']).item()
                    KL_total += data['KL'].item()
                    VAE_recon_total += data['VAE_recon'].item()
                    error_total += data['error'].item()
                else:
                    data = self.model(x)    # batch seq_len outsize
                    loss_mean = torch.mean(data['loss'])
                # Reward = loss_mean.data

                loss_mean.backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args['clip'])

                optimizer.step()


                print_loss_total += loss_mean.item()
                plot_loss_total += loss_mean.item()

                losses.append(loss_mean.item())

                # GPUtil.showUtilization()
                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    KL_avg = KL_total / print_every
                    KL_total = 0
                    CEloss_avg = CE_loss_total / print_every
                    CE_loss_total = 0
                    VAE_recon_avg = VAE_recon_total / print_every
                    VAE_recon_total = 0
                    error_avg = error_total / print_every
                    error_total = 0
                    print('%s (%d %d%%) loss = %.4f, CE_loss = %.4f, VAE recon = %.4f, KL = %.4f, error=%.4f' % (timeSince(start, iter / n_iters),
                                                 iter, iter / n_iters * 100, print_loss_avg, CEloss_avg, VAE_recon_avg, KL_avg, error_avg))
                    # GPUtil.showUtilization()
                    # del de_output,loss,true_mean
                    # GPUtil.showUtilization()
                    # torch.cuda.empty_cache()
                    # GPUtil.showUtilization()
                    if args['corpus'] != '1mb' or iter % 100000 == 0:
                        BLEU = self.Cal_BLEU_for_dataset('test')
                        print('Test ppl: ', BLEU)

                        if BLEU < min_BLEU or min_BLEU == -1:
                            print('BLEU = ', BLEU, '>= min_BLEU (', min_BLEU, '), saving model...')
                            min_BLEU = BLEU

                if iter % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0

                iter+=1

            # scheduler.step()
            BLEU = self.Cal_BLEU_for_dataset('test')
            if BLEU < min_BLEU or min_BLEU == -1:
                print('BLEU = ', BLEU, '>= min_BLEU (', min_BLEU, '), saving model...')
                torch.save(self.model, self.model_path.replace('model', 'model_'+args['LMtype']+'_'+args['corpus'] + '_'+str(args['maxLength'])+'_'+str(args['numLayers'])+'_'+str(args['embeddingSize'])))
                min_BLEU = BLEU

            print('Epoch ', epoch, 'loss = ', sum(losses) / len(losses), 'Valid BLEU = ', BLEU)

    def testMT(self):
        start = time.time()
        print('Test set BLEU = ', self.Cal_BLEU_for_dataset('test'))
        end = time.time()
        print('Test time: ', time.strftime("%H:%M:%S", time.gmtime(end-start )))

    def Cal_BLEU_for_dataset(self, datasetname):
        if not hasattr(self, 'testbatches' ):
            self.testbatches = {}
        if datasetname not in self.testbatches:
            self.testbatches[datasetname] = self.textData.getBatches(args['typename'], datasetname)
        self.model.eval()
        num = 0
        ave_loss = 0
        pred_ans = []  # bleu
        gold_ans = []
        rec = None
        with torch.no_grad():
            # print(len(self.testbatches[datasetname][0].decoderSeqs))
            for batch in self.testbatches[datasetname]:
                x = {}

                x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs))
                x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs))
                x['dec_len'] = batch.decoder_lens
                x['dec_target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs))

                decoded_words = self.model.predict(x)    # batch seq_len outsize
                pred_ans.extend(decoded_words)
                gold_ans.extend([[r] for r in batch.raw_source])

                if rec is None:
                    rec = (decoded_words[0], batch.raw_source[0], batch.raw_target[0])
                # GPUtil.showUtilization()
                # true_mean = recon_loss_mean
                # print(true_mean.size())
                # sum_true = data['true_mean'].sum().item()
                # ave_loss = (ave_loss * num + sum_true) / (num + len(data['true_mean']))
                #
                # num += len(data['true_mean'])


            print(rec)
            bleu = corpus_bleu(gold_ans, pred_ans)

        self.model.train()
        return bleu
 

if __name__ == '__main__':
    # args['corpus'] = 'wiki2'
    # args['LMtype'] = 'energy'
    args['norm_attn'] = True
    r = Runner()
    r.main()