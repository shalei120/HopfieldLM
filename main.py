# Copyright 2020 . All Rights Reserved.
# Author : Lei Sha
import functools
print = functools.partial(print, flush=True)
from LanguageModel import LanguageModel
from textdata_wiki2 import  TextData_wiki2
from textdata import  TextData_1mb
import time, sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import  tqdm
import time,os
import math,random
import nltk
from nltk.corpus import stopwords
import argparse

# import turibolt as bolt
import pickle
from Hyperparameters import args
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# import matplotlib.pyplot as plt
import numpy as np
import copy

# from kenLM import LMEvaluator as LMEr

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--batch', '-b')
parser.add_argument('--modelarch', '-m')
parser.add_argument('--data', '-d')

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

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

class Runner:
    def __init__(self):
        self.model_path = args['rootDir'] + '/model.pth'
        # self.model_bw_path = args['rootDir'] + '/model_bw.pth'
        self.drawcount = 0



    def main(self):
        if args['corpus'] == '1mb':
            self.textData =  TextData_1mb('LMbenchmark')
        elif  args['corpus'] == 'wiki2':
            self.textData = TextData_wiki2('LMbenchmark')
        # self.LMer = LMEr()
        self.start_token = self.textData.word2index['START_TOKEN']
        self.end_token = self.textData.word2index['END_TOKEN']
        args['vocabularySize'] = self.textData.getVocabularySize()
        print(self.textData.getVocabularySize())

        self.model = LanguageModel(self.textData.word2index, self.textData.index2word).to(args['device'])
        # self.model = torch.load(self.model_path.replace('model', 'model_'+'fw'), map_location=args['device'])
        params = sum([np.prod(p.size()) for p in self.model.parameters()])
        print(params, sum([sys.getsizeof(p.storage()) for p in self.model.parameters()])/1000000, 'm')
        self.trainLM()     # contain  model saving
        self.evaluateRandomly()
        self.testLM()

     

    def trainLM(self,  direction = 'fw', print_every=10000, plot_every=10, learning_rate=0.001):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        print(type(self.textData.word2index))

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=5.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        iter = 1
        batches = self.textData.getBatches()
        n_iters = len(batches)
        print('niters ',n_iters)

        args['trainseq2seq'] = False

        min_perplexity = -1

        self.Cal_perplexity_for_dataset('test', direction)

        for epoch in range(args['numEpochs']):
            losses = []

            for batch in batches:
                optimizer.zero_grad()
                x={}
                x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs))
                x['dec_len'] = batch.decoder_lens
                x['dec_target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs))


                # print(x['enc_input'][0],x['dec_input'][0],x['dec_target'][0])
                de_output, loss, true_mean = self.model(x)    # batch seq_len outsize

                loss_mean = torch.mean(loss)
                # Reward = loss_mean.data

                loss_mean.backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args['clip'])

                optimizer.step()

                print_loss_total += loss_mean.data
                plot_loss_total += loss_mean.data

                losses.append(loss_mean.data)

                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                                 iter, iter / n_iters * 100, print_loss_avg))

                    perplexity = self.Cal_perplexity_for_dataset('test', direction)
                    print('Test ppl: ', perplexity)

                if iter % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0

                iter+=1

            scheduler.step()
            perplexity = self.Cal_perplexity_for_dataset('test', direction)
            if perplexity < min_perplexity or min_perplexity == -1:
                print('perplexity = ', perplexity, '>= min_perplexity (', min_perplexity, '), saving model...')
                torch.save(self.model, self.model_path.replace('model', 'model_'+args['LMtype']+'_'+args['corpus'] + '_'+str(args['maxLength'])))
                min_perplexity = perplexity

            print('Epoch ', epoch, 'loss = ', sum(losses) / len(losses), 'Valid perplexity = ', perplexity)

        # self.test()
        # showPlot(plot_losses)

    def testLM(self):
        start = time.time()
        print('Test set BLEU = ', self.Cal_BLEU_for_dataset('test'))
        end = time.time()
        print('Test time: ', time.strftime("%H:%M:%S", time.gmtime(end-start )))

    def Cal_perplexity_for_dataset(self, datasetname, direction = 'fw'):
        if not hasattr(self, 'testbatches' ):
            self.testbatches = {}
        if datasetname not in self.testbatches:
            self.testbatches[datasetname] = self.textData.getBatches(datasetname)
        num = 0
        ave_loss = 0
        with torch.no_grad():
            for batch in self.testbatches[datasetname]:
                x = {}

                x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs))
                x['dec_len'] = batch.decoder_lens
                x['dec_target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs))

                de_output, recon_loss_mean, true_mean = self.model(x)
                # true_mean = recon_loss_mean
                # print(true_mean.size())
                ave_loss = (ave_loss * num + sum(true_mean)) / (num + len(true_mean))

                num += len(true_mean)

        return torch.exp(ave_loss)


    def indexesFromSentence(self,  sentence):
        return [self.textData.word2index[word] if word in self.textData.word2index else self.textData.word2index['UNK'] for word in sentence]

    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        # indexes.append(self.textData.word2index['END_TOKEN'])
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def evaluate(self,  sentence, correctlabel, max_length=20):
        with torch.no_grad():
            input_tensor = self.tensorFromSentence( sentence)
            input_length = input_tensor.size()[0]
            # encoder_hidden = encoder.initHidden()

            # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            x={}
            # print(input_tensor)
            x['enc_input'] = torch.transpose(input_tensor, 0,1)
            x['enc_len'] = [input_length]
            x['labels'] = [correctlabel]
            # print(x['enc_input'], x['enc_len'])
            # print(x['enc_input'].shape)
            decoded_words, label,_ = self.model.predict(x, True)

            return decoded_words, label

    def evaluateRandomly(self, n=10):
        for i in range(n):
            sample = random.choice(self.textData.datasets['train'])
            print('>', sample)
            output_words, label = self.evaluate(sample[2], sample[1])
            output_sentence = ' '.join(output_words[0])  # batch=1
            print('<', output_sentence, label)
            print('')

 

if __name__ == '__main__':
    args['corpus'] = 'wiki2'
    args['LMtype'] = 'energy'
    r = Runner()
    # r.textData = TextData('LMbenchmark')

    # r.Get_KeyWords(3000)
    r.main()