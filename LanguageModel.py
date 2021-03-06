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

from textdata import TextData
from kenLM import LMEvaluator as LMEr

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




    def forward(self, x):
        self.decoderInputs = x['dec_input'].to(self.device)
        self.decoder_lengths = x['dec_len']
        self.decoderTargets = x['dec_target'].to(self.device)

        batch_size = self.decoderInputs.size()[0]
        self.dec_len = self.decoderInputs.size()[1]


        dec_input_embed = self.embedding(self.decoderInputs).to(self.device)

        init_state = (self.init_state[0].repeat([1, batch_size, 1]), self.init_state[1].repeat([1, batch_size, 1]))

        de_outputs, de_state = self.decoder_t(init_state, dec_input_embed, batch_size, self.dec_len)

        recon_loss = self.CEloss(torch.transpose(de_outputs, 1, 2), self.decoderTargets)
        mask = torch.sign(self.decoderTargets.float())
        recon_loss = torch.squeeze(recon_loss) * mask

        recon_loss_mean = torch.mean(recon_loss, dim=-1)
        return de_outputs, recon_loss_mean

    def predict(self, x):
        decoderInputs = x['dec_input'].to(self.device)
        decoderTarget = x['dec_target'].to(self.device)
        batch_size = decoderInputs.size()[0]

        init_state = (torch.rand(args['dec_numlayer'], batch_size, args['hiddenSize'], device=self.device),
                           torch.rand(args['dec_numlayer'], batch_size, args['hiddenSize'], device=self.device))

        dec_len = decoderInputs.size()[1]
        dec_input_embed = self.embedding(decoderInputs).to(self.device)

        de_outputs, de_state = self.decoder_t(init_state, dec_input_embed, batch_size, dec_len)

        de_outputs = self.logsoftmax(de_outputs)

        recon_loss = self.CEloss(torch.transpose(de_outputs, 1, 2), decoderTarget)
        mask = torch.sign(decoderTarget.float())
        recon_loss = torch.squeeze(recon_loss) * mask

        recon_loss_mean = torch.mean(recon_loss, dim=-1)

        return de_outputs, recon_loss_mean

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(args['device'])
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature = args['temperature']):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard



  
   

    def generate(self, init_state, senlen, assistseq = [], startseq = []):
        self.batch_size = 1
        init_state = (init_state[0].repeat([1, self.batch_size, 1]), init_state[1].repeat([1, self.batch_size, 1]))

        de_words = self.decoder_g(init_state, senlen, assistseq, startseq)

        return de_words

    def generate_sample(self, init_state, senlen, sample_n = 16):

        init_state = (init_state[0].repeat([1, sample_n, 1]), init_state[1].repeat([1, sample_n, 1]))

        de_words_vec, de_words = self.decoder_g_sample(init_state, senlen, sample_n)

        return de_words_vec, de_words

    def generate_beam(self, en_state):
        de_words = self.decoder_g_beam(en_state)
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

    def decoder_g(self, initial_state, senlen, assistseq = [], startseq = []):
        state = initial_state
        # sentence_emb = sentence_emb.view(self.batch_size,1, -1 )

        decoded_words = []
        decoder_input_id = torch.tensor([[self.word2index['START_TOKEN'] for _ in range(self.batch_size)]], device=args['device'])  # SOS 1*batch
        decoder_input = self.embedding(decoder_input_id).contiguous().to(args['device'])
        # print('decoder input: ', decoder_input.shape)
        decoder_id_res = []


        if len(assistseq) > 0:
            decoder_input = assistseq[0].unsqueeze(0).unsqueeze(0).to(args['device'])
            # print('decoder input2: ', decoder_input.shape)
        if len(startseq) > 0:
            decoder_output, state = self.dec_unit(decoder_input, state)
            startseq_input = self.embedding(torch.LongTensor(startseq)).unsqueeze(1).to(args['device']) #  seq batch emb
            for ind in range(len(startseq)-1):
                input_word = startseq_input[ind,:,:].unsqueeze(0)
                decoder_output, state = self.dec_unit(input_word, state)

            decoder_input = startseq_input[-1,:,:].unsqueeze(0)

        for di in range(senlen):

            # decoder_output, state = self.dec_unit(torch.cat([decoder_input, sentence_emb], dim = -1), state)
            decoder_output, state = self.dec_unit(decoder_input, state)

            decoder_output = self.out_unit(decoder_output)
            decoder_output = self.symbol_prob_set_zero(decoder_output)
            topv, topi = decoder_output.data.topk(1, dim = -1)

            decoder_input_id = topi[:,:,0].detach()
            decoder_id_res.append(decoder_input_id)
            decoder_input = self.embedding(decoder_input_id).to(args['device'])

        decoder_id_res = torch.cat(decoder_id_res, dim = 0)  #seqlen * batch

        for b in range(self.batch_size):
            decode_id_list = list(decoder_id_res[:,b])
            if self.word2index['END_TOKEN'] in decode_id_list:
                decode_id_list = decode_id_list[:decode_id_list.index(self.word2index['END_TOKEN'])] if decode_id_list[0] != self.word2index['END_TOKEN'] else [self.word2index['END_TOKEN']]
            decoded_words.append([self.index2word[id] for id in decode_id_list])

        decode_id_list = startseq +[ int(n) for n in  decode_id_list]

        return decode_id_list

    def decoder_g_sample(self, initial_state, senlen, sample_n = 10):
        state = initial_state
        # sentence_emb = sentence_emb.view(self.batch_size,1, -1 )

        decoded_words = []
        decoder_input_id = torch.tensor([[self.word2index['START_TOKEN'] for _ in range(sample_n)]], device=args['device'])  # SOS 1*batch
        decoder_input = self.embedding(decoder_input_id).contiguous().to(args['device'])
        # print('decoder input: ', decoder_input.shape)
        decoder_vec_res = []
        decoder_id_res = []


        for di in range(senlen):

            # decoder_output, state = self.dec_unit(torch.cat([decoder_input, sentence_emb], dim = -1), state)
            decoder_output, state = self.dec_unit(decoder_input, state)

            decoder_output = self.out_unit(decoder_output)
            decoder_output = self.symbol_prob_set_zero(decoder_output)
            decoder_sample = self.gumbel_softmax(decoder_output)
            decoder_output_ids = torch.argmax(decoder_sample, dim = -1) # seq batch

            decoder_input = torch.einsum('sbv,ve->sbe', decoder_sample, self.embedding.weight)
            decoder_vec_res.append(decoder_input)
            decoder_id_res.append(decoder_output_ids)

        decoder_vec_res_cat = torch.cat(decoder_vec_res, dim = 0)  #seqlen * batch * embsize
        decoder_vec_res_cat = torch.transpose(decoder_vec_res_cat, 0,1)

        decoder_id_res_cat = torch.cat(decoder_id_res, dim = 0) # seq batch
        decoder_id_res_cat = torch.transpose(decoder_id_res_cat, 0,1)

        return decoder_vec_res_cat, decoder_id_res_cat

    def decoder_g_beam(self, initial_state, beam_width = 10):
        parent = self
        class Subseq:
            def __init__(self ):
                self.logp = 0.0
                self.sequence = [parent.word2index['START_TOKEN']]

            def append(self, wordindex, logp):
                self.sequence.append(wordindex)
                self.logp += logp

            def append_createnew(self, wordindex, logp):
                newss = copy.deepcopy(self)
                newss.sequence.append(wordindex)
                newss.logp += logp
                return newss

            def eval(self):
                return self.logp / float(len(self.sequence) - 1 + 1e-6)

            def __lt__(self, other): # add negative
                return self.eval() > other.eval()

        state = initial_state
        pq = PriorityQueue()
        pq.put(Subseq())

        for di in range(self.max_length):
            pqitems = []
            for _ in range(beam_width):
                pqitems.append(pq.get())
                if pq.empty():
                    break
            pq.queue.clear()

            end = True
            for subseq in pqitems:
                if subseq.sequence[-1] == self.word2index['END_TOKEN']:
                    pq.put(subseq)
                else:
                    end = False
                    lastindex = subseq.sequence[-1]
                    decoder_input_id = torch.tensor([[lastindex]], device=self.device)  # SOS
                    decoder_input = self.embedding(decoder_input_id).contiguous().to(self.device)
                    decoder_output, state = self.dec_unit(decoder_input, state)

                    decoder_output = self.out_unit(decoder_output)
                    decoder_output = self.logsoftmax(decoder_output)

                    logps, indexes = decoder_output.data.topk(beam_width)

                    for logp, index in zip(logps[0][0], indexes[0][0]):
                        newss = subseq.append_createnew(index.item(), logp)
                        pq.put(newss )
            if end:
                break

        finalseq = pq.get()

        decoded_words = []
        for i in finalseq.sequence[1:]:
            decoded_words.append(self.index2word[i])

        return decoded_words

    def predict_next(self, sentence_emb):
        # print(type(sentence_emb))
        # print(sentence_emb.size)
        batch_size = sentence_emb.size()[0]
        dec_len = sentence_emb.size()[1]
        dec_input_embed = sentence_emb.to(args['device'])

        init_state = (self.init_state[0].repeat([1, batch_size, 1]), self.init_state[1].repeat([1, batch_size, 1]))

        de_outputs, de_state = self.decoder_t(init_state, dec_input_embed, batch_size, dec_len) # batch len emb

        nextword_prob = self.softmax(de_outputs[:, -1, :])

        return nextword_prob

    def predict_next_batch(self, sentence_emb, senlen):
        # print(type(sentence_emb))
        # print(sentence_emb.size)
        batch_size = sentence_emb.size()[0]
        dec_len = sentence_emb.size()[1]
        dec_input_embed = sentence_emb.to(args['device'])

        init_state = (self.init_state[0].repeat([1, batch_size, 1]), self.init_state[1].repeat([1, batch_size, 1]))

        de_outputs, de_state = self.decoder_t(init_state, dec_input_embed, batch_size, dec_len) # batch len emb

        de_real_output = de_outputs.gather(1, senlen.view(batch_size, 1, 1).repeat([1,1,args['vocabularySize']]) - 1) # batch  1 voc

        nextword_prob = self.softmax(de_real_output[:,0,:]) # batch voc

        return nextword_prob

    def symbol_prob_set_zero(self, voc_probs):
        '''
        :param voc_probs:  batch seq vocsize
        :return:
        '''

        if not hasattr(self, 'symbols_index'):
            self.symbols_index = []
            with open(args['rootDir'] + '/symbol_index.txt','r') as handle:
                line = handle.readline()
                self.symbols_index = [int(ind) for ind in line.strip().split()]

        voc_probs[:,:,0] = 0
        voc_probs[:,:,3] = 0
        for ind in self.symbols_index:
            voc_probs[:, :, ind] = 0

        return voc_probs


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



