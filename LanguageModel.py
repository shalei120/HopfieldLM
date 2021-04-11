import functools
print = functools.partial(print, flush=True)
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import numpy as np

import datetime
from Hyperparameters import args
from queue import PriorityQueue
import copy

from kenLM import LMEvaluator as LMEr

from modules import Hopfield, HopfieldPooling, HopfieldLayer
from modules.transformer import  HopfieldEncoderLayer
from Transformer import EnergyTransformerEncoderLayer
class LanguageModel(nn.Module):
    def __init__(self,w2i, i2w):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(LanguageModel, self).__init__()
        print("LanguageModel creation...")

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']
        self.device = args['device']
        self.batch_size = args['batchSize']

        self.dtype = 'float32'

        self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize']).to(self.device)

        if args['decunit'] == 'lstm':
            self.dec_unit = nn.LSTM(input_size=args['embeddingSize'],
                                    hidden_size=args['hiddenSize'],
                                    num_layers=args['dec_numlayer']).to(self.device)
        elif args['decunit'] == 'gru':
            self.dec_unit = nn.GRU(input_size=args['embeddingSize'],
                                   hidden_size=args['hiddenSize'],
                                   num_layers=args['dec_numlayer']).to(self.device)

        self.out_unit = nn.Linear(args['hiddenSize'], args['vocabularySize']).to(self.device)
        self.logsoftmax = nn.LogSoftmax(dim = -1)

        self.element_len = args['hiddenSize']

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
        self.CEloss = torch.nn.CrossEntropyLoss(reduction='none')

        self.init_state = (torch.rand(args['dec_numlayer'], 1, args['hiddenSize'], device=self.device),
                           torch.rand(args['dec_numlayer'], 1, args['hiddenSize'], device=self.device))

        if args['LMtype'] == 'asso':
            self.hopfield = Hopfield(
                input_size=args['embeddingSize'] )
            output_projection = nn.Linear(in_features=self.hopfield.output_size, out_features=args['vocabularySize'])
            self.hp_network = nn.Sequential(self.hopfield, output_projection).to(self.device)
        elif args['LMtype'] == 'asso_enco':
            self.hopfield = Hopfield(
                input_size=args['embeddingSize'] )
            self.hp_network = HopfieldEncoderLayer(self.hopfield)
            self.output_projection = nn.Linear(in_features=args['embeddingSize'], out_features=args['vocabularySize'])
        elif args['LMtype'] == 'transformer':
            self.trans_net = nn.TransformerEncoderLayer(d_model=args['embeddingSize'], nhead=1).to(self.device)
            self.transformer_encoder = nn.TransformerEncoder(self.trans_net, num_layers=1).to(self.device)
            self.output_projection = nn.Linear(in_features=args['embeddingSize'], out_features=args['vocabularySize'])
            # self.transformer_network = nn.Sequential(self.transformer_encoder, output_projection).to(self.device)
        elif args['LMtype'] == 'energy':
            self.trans_net = EnergyTransformerEncoderLayer(d_model=args['embeddingSize'], nhead=1).to(self.device)
            # self.transformer_encoder = nn.TransformerEncoder(self.trans_net, num_layers=1).to(self.device)
            self.output_projection = nn.Linear(in_features=args['embeddingSize'], out_features=args['vocabularySize']).to(self.device)
            # self.transformer_network = nn.Sequential(self.transformer_encoder, output_projection).to(self.device)


    def generate_square_subsequent_mask(self, sz):
        mask = torch.logical_not(torch.triu(torch.ones(sz, sz)) == 1)
        mask[0, 0] = True
        # mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



    def build(self, x):
        self.decoderInputs = x['dec_input'].to(self.device)
        self.decoder_lengths = x['dec_len']
        self.decoderTargets = x['dec_target'].to(self.device)

        batch_size = self.decoderInputs.size()[0]
        self.dec_len = self.decoderInputs.size()[1]
        dec_input_embed = self.embedding(self.decoderInputs)

        if args['LMtype']== 'lstm':
            init_state = (self.init_state[0].repeat([1, batch_size, 1]), self.init_state[1].repeat([1, batch_size, 1]))
            de_outputs, de_state = self.decoder_t(init_state, dec_input_embed, batch_size, self.dec_len)
        elif args['LMtype']== 'asso':
            # print(args['maxLengthDeco'], dec_input_embed.size())
            de_outputs = self.hp_network(dec_input_embed)
        elif args['LMtype']== 'asso_enco':
            src_mask = self.generate_square_subsequent_mask(self.dec_len).to(self.device)
            print(args['maxLengthDeco'], dec_input_embed.size(), src_mask.size())
            de_outputs = self.hp_network(dec_input_embed,src_mask)
            de_outputs = self.output_projection(de_outputs)
            # de_outputs = de_outputs.transpose(0,1)
        elif args['LMtype']== 'transformer':
            src_mask = self.generate_square_subsequent_mask(self.dec_len).to(self.device)
            de_outputs = self.transformer_encoder(dec_input_embed.transpose(0,1), mask=src_mask)
            de_outputs = self.output_projection(de_outputs)
            de_outputs = de_outputs.transpose(0,1)
        elif args['LMtype'] == 'energy':
            src_mask = self.generate_square_subsequent_mask(self.dec_len).to(self.device)
            de_outputs = self.trans_net(dec_input_embed, src_mask = src_mask).to(self.device)
            de_outputs = self.output_projection(de_outputs)
            # de_outputs = de_outputs.transpose(0,1)
        # print(de_outputs.size(),self.decoderTargets.size())
        # print(de_outputs.size(),self.decoderTargets.size() )
        recon_loss = self.CEloss(torch.transpose(de_outputs, 1, 2), self.decoderTargets)
        mask = torch.sign(self.decoderTargets.float())
        recon_loss = torch.squeeze(recon_loss) * mask

        recon_loss_mean = torch.mean(recon_loss, dim=-1)
        # print(recon_loss.size(), mask.size())
        true_mean = recon_loss.sum(1) / mask.sum(1)
        return de_outputs, recon_loss_mean, true_mean

    def forward(self, x):
        de_outputs, recon_loss_mean, true_mean = self.build(x)
        return de_outputs, recon_loss_mean, true_mean


    def generate(self, init_state, senlen, assistseq = [], startseq = []):
        self.batch_size = 1
        init_state = (init_state[0].repeat([1, self.batch_size, 1]), init_state[1].repeat([1, self.batch_size, 1]))

        de_words = self.decoder_g(init_state, senlen, assistseq, startseq)

        return de_words

    def decoder_t(self, initial_state, inputs, batch_size, dec_len):
        inputs = torch.transpose(inputs, 0,1).contiguous()
        state = initial_state

        output, out_state = self.dec_unit(inputs, state)
        # output = output.cpu()

        output = self.out_unit(output.view(batch_size * dec_len, args['hiddenSize']))
        output = output.view(dec_len, batch_size, args['vocabularySize'])
        output = torch.transpose(output, 0,1)
        return output, out_state



if __name__ == '__main__':
    LMer = LMEr()
    textData = TextData('LMbenchmark')
    # args['vocabularySize'] = textData.getVocabularySize()
    # model_path = args['rootDir'] + '/model.pth'
    # model = torch.load(model_path, map_location=args['device'])
    # sentoken =  torch.LongTensor([[textData.word2index[w] for w in 'START_TOKEN three of hit by the companies unite the a the next END_TOKEN'.split()]])
    # y_hat_emb = model.embedding(sentoken.to(args['device']))
    # loss = model.Cal_LMloss_with_fuzzyword(y_hat_emb, torch.LongTensor(sentoken).to(args['device']))
    # loss2 = model.Cal_sen_LMloss(y_hat_emb, torch.LongTensor(sentoken).to(args['device']))
    # print(torch.exp(loss), torch.exp(loss2), loss)
    # print(LMer.Perplexity(['three of hit by the companies unite the a the next']))



