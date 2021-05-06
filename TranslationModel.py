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

# from kenLM import LMEvaluator as LMEr

from modules import Hopfield, HopfieldPooling, HopfieldLayer
from modules.transformer import  HopfieldEncoderLayer
from Transformer import EnergyTransformerEncoderLayer,EnergyTransformerEncoder
class TranslationModel(nn.Module):
    def __init__(self,w2i, i2w):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(TranslationModel, self).__init__()
        print("TranslationModel creation...")

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']
        self.device = args['device']
        self.batch_size = args['batchSize']

        self.dtype = 'float32'

        self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize']).to(self.device)

        # if args['decunit'] == 'lstm':
        #     self.dec_unit = nn.LSTM(input_size=args['embeddingSize'],
        #                             hidden_size=args['hiddenSize'],
        #                             num_layers=args['dec_numlayer']).to(self.device)
        # elif args['decunit'] == 'gru':
        #     self.dec_unit = nn.GRU(input_size=args['embeddingSize'],
        #                            hidden_size=args['hiddenSize'],
        #                            num_layers=args['dec_numlayer']).to(self.device)

        # self.out_unit = nn.Linear(args['hiddenSize'], args['vocabularySize']).to(self.device)
        self.logsoftmax = nn.LogSoftmax(dim = -1)

        self.element_len = args['hiddenSize']

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
        self.CEloss = torch.nn.CrossEntropyLoss(reduction='none')

        # self.init_state = (torch.rand(args['dec_numlayer'], 1, args['hiddenSize'], device=self.device),
        #                    torch.rand(args['dec_numlayer'], 1, args['hiddenSize'], device=self.device))

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
            self.trans_net = nn.TransformerEncoderLayer(d_model=args['embeddingSize'], nhead=args['nhead']).to(self.device)
            self.transformer_encoder = nn.TransformerEncoder(self.trans_net, num_layers=args['numLayers']).to(self.device)
            self.trans_de_net = nn.TransformerDecoderLayer(d_model=args['embeddingSize'], nhead=args['nhead']).to(self.device)
            self.transformer_decoder = nn.TransformerDecoder(self.trans_de_net, num_layers=args['numLayers']).to(self.device)
            self.output_projection = nn.Linear(in_features=args['embeddingSize'], out_features=args['vocabularySize'])
            # self.transformer_network = nn.Sequential(self.transformer_encoder, output_projection).to(self.device)
        elif args['LMtype'] == 'energy':
            self.trans_net = EnergyTransformerEncoderLayer(d_model=args['embeddingSize'], nhead=args['nhead']).to(self.device)
            self.energytransformer_encoder = EnergyTransformerEncoder(self.trans_net, num_layers=args['numLayers']).to(self.device)
            self.output_projection = nn.Linear(in_features=args['embeddingSize'], out_features=args['vocabularySize']).to(self.device)
            # self.transformer_network = nn.Sequential(self.transformer_encoder, output_projection).to(self.device)
            # self.energytransformer_encoder_neg = EnergyTransformerEncoder(self.trans_net, num_layers=args['numLayers'], choice = 0).to(self.device)
            # self.output_projection_neg = nn.Linear(in_features=args['embeddingSize'], out_features=args['vocabularySize']).to(self.device)


    def generate_square_subsequent_mask(self, sz):
        # mask = torch.logical_not(torch.triu(torch.ones(sz, sz)) == 1)
        # mask[0, 0] = True
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



    def build(self, x, training):
        self.encoderInputs = x['enc_input'].to(self.device)
        self.decoderInputs = x['dec_input'].to(self.device)
        self.decoder_lengths = x['dec_len']
        self.decoderTargets = x['dec_target'].to(self.device)

        batch_size = self.decoderInputs.size()[0]
        self.dec_len = self.decoderInputs.size()[1]
        enc_input_embed = self.embedding(self.encoderInputs)
        dec_input_embed = self.embedding(self.decoderInputs)
        mask = torch.sign(self.decoderTargets.float())

        if args['LMtype']== 'lstm':
            init_state = (self.init_state[0].repeat([1, batch_size, 1]), self.init_state[1].repeat([1, batch_size, 1]))
            de_outputs, de_state = self.decoder_t(init_state, dec_input_embed, batch_size, self.dec_len)
        elif args['LMtype']== 'asso':
            # print(args['maxLengthDeco'], dec_input_embed.size())
            de_outputs = self.hp_network(dec_input_embed)
        elif args['LMtype']== 'asso_enco':
            src_mask = self.generate_square_subsequent_mask(self.dec_len).to(self.device)
            # print(args['maxLengthDeco'], dec_input_embed.size(), src_mask.size())
            de_outputs = self.hp_network(dec_input_embed,src_mask)
            de_outputs = self.output_projection(de_outputs)
            # de_outputs = de_outputs.transpose(0,1)
        elif args['LMtype']== 'transformer':
            src_mask = self.generate_square_subsequent_mask(self.dec_len).to(self.device)
            enc_hid = self.transformer_encoder(enc_input_embed.transpose(0,1))
            de_outputs = self.transformer_decoder(dec_input_embed.transpose(0,1), enc_hid, tgt_mask=src_mask)
            de_outputs = self.output_projection(de_outputs)
            de_outputs = de_outputs.transpose(0,1)
            # print(de_outputs.size())
        elif args['LMtype'] == 'energy':
            src_mask = self.generate_square_subsequent_mask(self.dec_len).to(self.device)
            _, loss_tuple, error, de_outputs_list = self.energytransformer_encoder(dec_input_embed, mask = src_mask, src_key_padding_mask = mask, training=training)
            de_outputs = self.output_projection(de_outputs_list[-1])
            # de_outputs_neg, loss_tuple_neg, error_neg = self.energytransformer_encoder_neg(dec_input_embed, mask = src_mask, src_key_padding_mask = mask, training=training)
            # de_outputs_neg = self.output_projection(de_outputs_neg)
            # de_outputs = de_outputs.transpose(0,1)
        # print(de_outputs.size(),self.decoderTargets.size())
        # print(de_outputs.size(),self.decoderTargets.size() )
        recon_loss = self.CEloss(torch.transpose(de_outputs, 1, 2), self.decoderTargets)
        recon_loss = torch.squeeze(recon_loss) * mask
        recon_loss_mean = torch.mean(recon_loss, dim=-1)


        # recon_loss_neg = self.CEloss(torch.transpose(de_outputs_neg, 1, 2), self.decoderTargets)
        # recon_loss_neg = torch.squeeze(recon_loss_neg) * mask
        # recon_loss_mean_neg = torch.mean(recon_loss_neg, dim=-1)
        # print(recon_loss.size(), mask.size())
        true_mean = recon_loss.sum(1) / mask.sum(1)

        data = {'de_outputs': de_outputs,
                'loss':recon_loss_mean,
                'true_mean':true_mean}

        if args['LMtype'] == 'energy':
            targetvector = self.embedding(self.decoderTargets)
            data['KL'] = loss_tuple[1]
            data['VAE_recon']  = loss_tuple[0]
            # recon_loss_mean = recon_loss_mean.mean() + 100*KL
            # c_loss = 0
            # for out in de_outputs_list[:-1]:
            #     dots = torch.einsum('bse,bse->bs',out, targetvector)
            #     # dots1 = torch.exp(-dots.clone())
            #     dots *= mask
            #     c_loss += torch.exp(-dots.mean())

            data['error'] = error
            # self.decoder = self.en
        elif args['LMtype'] == 'transformer':
            data['enc_output'] = enc_hid.transpose(0,1)
            self.decoder = self.transformer_decoder



        return data

    def forward(self, x):
        data = self.build(x, training=True)
        return data

    def predict(self, x):
        bs = x['dec_target'].size()[0]
        data = self.build(x, training=False)
        # for a single batch x
        encoder_output = data['enc_output']  # (bs, input_len, d_model)

        decoded_words = []
        # initialized the input of the decoder with sos_idx (start of sentence token idx)
        output = torch.ones(bs, self.max_length).long().to(args['device']) * self.word2index['START_TOKEN']
        for t in range(1, self.max_length):
            tgt_emb = self.embedding(output[:, :t]).transpose(0, 1)
            # tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(
            #     t).to(device).transpose(0, 1)
            decoder_output = self.decoder(tgt=tgt_emb,
                                     memory=encoder_output,
                                     tgt_mask=None)

            pred_proba_t = self.output_projection(decoder_output)[-1, :, :]
            output_t = pred_proba_t.data.topk(1)[1].squeeze()
            output[:, t] = output_t
            # print(output)

        for b in range(bs):
            decode_id_list = list(output[b, :])
            # print(decode_id_list)
            if self.word2index['END_TOKEN'] in decode_id_list:
                decode_id_list = decode_id_list[1:decode_id_list.index(self.word2index['END_TOKEN'])] \
                    if decode_id_list[0] != self.word2index['END_TOKEN'] else [self.word2index['END_TOKEN']]
            decoded_words.append([self.index2word[id] for id in decode_id_list])
            # print(decoded_words)
        return decoded_words




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



