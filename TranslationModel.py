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
from EnergyTransformer import EnergyTransformerEncoderLayer,EnergyTransformerEncoder
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
        self.batch_size = args['batchSize']

        self.dtype = 'float32'

        self.embedding_src = nn.Embedding(args['vocabularySize_src'], args['embeddingSize']).to(args['device'])
        self.embedding_tgt = nn.Embedding(args['vocabularySize_tgt'], args['embeddingSize']).to(args['device'])

        # if args['decunit'] == 'lstm':
        #     self.dec_unit = nn.LSTM(input_size=args['embeddingSize'],
        #                             hidden_size=args['hiddenSize'],
        #                             num_layers=args['dec_numlayer']).to(args['device'])
        # elif args['decunit'] == 'gru':
        #     self.dec_unit = nn.GRU(input_size=args['embeddingSize'],
        #                            hidden_size=args['hiddenSize'],
        #                            num_layers=args['dec_numlayer']).to(args['device'])

        # self.out_unit = nn.Linear(args['hiddenSize'], args['vocabularySize']).to(args['device'])
        self.logsoftmax = nn.LogSoftmax(dim = -1)

        self.element_len = args['hiddenSize']

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
        self.CEloss = torch.nn.CrossEntropyLoss(reduction='none')

        # self.init_state = (torch.rand(args['dec_numlayer'], 1, args['hiddenSize'], device=args['device']),
        #                    torch.rand(args['dec_numlayer'], 1, args['hiddenSize'], device=args['device']))

        if args['LMtype'] == 'asso':
            self.hopfield = Hopfield(
                input_size=args['embeddingSize'] )
            output_projection = nn.Linear(in_features=self.hopfield.output_size, out_features=args['vocabularySize_tgt'])
            self.hp_network = nn.Sequential(self.hopfield, output_projection).to(args['device'])
        elif args['LMtype'] == 'asso_enco':
            self.hopfield = Hopfield(
                input_size=args['embeddingSize'] )
            self.hp_network = HopfieldEncoderLayer(self.hopfield)
            self.output_projection = nn.Linear(in_features=args['embeddingSize'], out_features=args['vocabularySize_tgt'])
        elif args['LMtype'] == 'transformer':
            self.trans_net = nn.TransformerEncoderLayer(d_model=args['embeddingSize'], dim_feedforward = 1024, nhead=args['nhead']).to(args['device'])
            self.transformer_encoder = nn.TransformerEncoder(self.trans_net, num_layers=args['numLayers']).to(args['device'])
            self.trans_de_net = nn.TransformerDecoderLayer(d_model=args['embeddingSize'],dim_feedforward = 1024,nhead=args['nhead']).to(args['device'])
            self.transformer_decoder = nn.TransformerDecoder(self.trans_de_net, num_layers=args['numLayers']).to(args['device'])
            self.output_projection = nn.Linear(in_features=args['embeddingSize'], out_features=args['vocabularySize_tgt'])
            # self.transformer_network = nn.Sequential(self.transformer_encoder, output_projection).to(args['device'])
        elif args['LMtype'] == 'energy':
            self.trans_net = EnergyTransformerEncoderLayer(d_model=args['embeddingSize'], nhead=args['nhead']).to(args['device'])
            self.energytransformer_encoder = EnergyTransformerEncoder(self.trans_net, num_layers=args['numLayers']).to(args['device'])
            self.output_projection = nn.Linear(in_features=args['embeddingSize'], out_features=args['vocabularySize_tgt']).to(args['device'])
            # self.transformer_network = nn.Sequential(self.transformer_encoder, output_projection).to(args['device'])
            # self.energytransformer_encoder_neg = EnergyTransformerEncoder(self.trans_net, num_layers=args['numLayers'], choice = 0).to(args['device'])
            # self.output_projection_neg = nn.Linear(in_features=args['embeddingSize'], out_features=args['vocabularySize']).to(args['device'])

        self.sequence_generator = self.build_generator(
            [model], Namespace(**gen_args)
        )

    def generate_square_subsequent_mask(self, sz):
        # mask = torch.logical_not(torch.triu(torch.ones(sz, sz)) == 1)
        # mask[0, 0] = True
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



    def build(self, x, training, smooth_epsilon = 0.1):
        self.encoderInputs = x['enc_input'].to(args['device'])
        self.decoderInputs = x['dec_input'].to(args['device'])
        self.decoder_lengths = x['dec_len']
        self.decoderTargets = x['dec_target'].to(args['device'])

        # print(self.encoderInputs[0], self.decoderInputs[0], self.decoderTargets[0])

        batch_size = self.decoderInputs.size()[0]
        self.dec_len = self.decoderInputs.size()[1]
        enc_input_embed = self.embedding_src(self.encoderInputs)
        dec_input_embed = self.embedding_tgt(self.decoderInputs)
        mask = torch.sign(self.decoderTargets.float())

        if args['LMtype']== 'lstm':
            init_state = (self.init_state[0].repeat([1, batch_size, 1]), self.init_state[1].repeat([1, batch_size, 1]))
            de_outputs, de_state = self.decoder_t(init_state, dec_input_embed, batch_size, self.dec_len)
        elif args['LMtype']== 'asso':
            # print(args['maxLengthDeco'], dec_input_embed.size())
            de_outputs = self.hp_network(dec_input_embed)
        elif args['LMtype']== 'asso_enco':
            src_mask = self.generate_square_subsequent_mask(self.dec_len).to(args['device'])
            # print(args['maxLengthDeco'], dec_input_embed.size(), src_mask.size())
            de_outputs = self.hp_network(dec_input_embed,src_mask)
            de_outputs = self.output_projection(de_outputs)
            # de_outputs = de_outputs.transpose(0,1)
        elif args['LMtype']== 'transformer':
            src_mask = self.generate_square_subsequent_mask(self.dec_len).to(args['device'])
            enc_hid = self.transformer_encoder(enc_input_embed.transpose(0,1))
            de_outputs = self.transformer_decoder(dec_input_embed.transpose(0,1), enc_hid, tgt_mask=src_mask)
            de_outputs = self.output_projection(de_outputs)
            de_outputs = de_outputs.transpose(0,1) # b s e
            # print(de_outputs.size())
        elif args['LMtype'] == 'energy':
            src_mask = self.generate_square_subsequent_mask(self.dec_len).to(args['device'])
            _, loss_tuple, error, de_outputs_list = self.energytransformer_encoder(dec_input_embed, mask = src_mask, src_key_padding_mask = mask, training=training)
            de_outputs = self.output_projection(de_outputs_list[-1])
            # de_outputs_neg, loss_tuple_neg, error_neg = self.energytransformer_encoder_neg(dec_input_embed, mask = src_mask, src_key_padding_mask = mask, training=training)
            # de_outputs_neg = self.output_projection(de_outputs_neg)
            # de_outputs = de_outputs.transpose(0,1)
        # print(de_outputs.size(),self.decoderTargets.size())
        # print(de_outputs.size(),self.decoderTargets.size() )
        recon_loss = self.CEloss(torch.transpose(de_outputs, 1, 2), self.decoderTargets)
        recon_loss = torch.squeeze(recon_loss) * mask
        recon_loss_mean = torch.sum(recon_loss, dim = -1)

        lprobs = - F.log_softmax(de_outputs) # b s v
        smooth_loss = lprobs.sum(-1)
        smooth_loss = torch.sum(smooth_loss * mask,dim = -1)

        eps_i = smooth_epsilon / (lprobs.size(-1) - 1)

        # recon_loss_neg = self.CEloss(torch.transpose(de_outputs_neg, 1, 2), self.decoderTargets)
        # recon_loss_neg = torch.squeeze(recon_loss_neg) * mask
        # recon_loss_mean_neg = torch.mean(recon_loss_neg, dim=-1)
        # print(recon_loss.size(), mask.size())
        true_mean = recon_loss.sum(1) / mask.sum(1)

        data = {'de_outputs': de_outputs,
                'loss':((1.0 - smooth_epsilon - eps_i) * recon_loss_mean + eps_i * smooth_loss).mean(),
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
        sample={
            'id': x['id'],
            'nsentences':len(x['id']),
            'ntokens':-1,
            'net_input':{
                'src_tokens':x['enc_input'],
                'src_lengths':torch.sign(x['enc_input'].float()).sum(1) ,
                'prev_output_tokens':x['dec_input'],
            },
            'target':x['dec_target']
        }
        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)

    def predict_ori(self, x):
        bs = x['dec_target'].size()[0]
        data = self.build(x, training=False)
        # for a single batch x
        # encoder_output = data['enc_output']  # (bs, input_len, d_model)
        #
        decoded_words = []
        # # initialized the input of the decoder with sos_idx (start of sentence token idx)
        output = torch.ones(bs, self.max_length).long().to(args['device']) * self.word2index['START_TOKEN']
        for t in range(1, self.max_length):
            tgt_emb = self.embedding_tgt(output[:, :t]).transpose(0, 1)
            # tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(
            #     t).to(device).transpose(0, 1)
            decoder_output = self.decoder(tgt=tgt_emb,
                                     memory=data['enc_output'],
                                     tgt_mask=None)  # s b e

            pred_proba_t = self.output_projection(decoder_output)[-1, :, :]
            output_t = pred_proba_t.data.topk(1)[1].squeeze()
            output[:, t] = output_t
            # print(output)
        # output = torch.argmax(data['de_outputs'], dim= 2)
        # print(data['de_outputs'].size())
        for b in range(bs):
            decode_id_list = list(output[b, :])
            # print(decode_id_list)
            if self.word2index['END_TOKEN'] in decode_id_list:
                decode_id_list = decode_id_list[1:decode_id_list.index(self.word2index['END_TOKEN'])] \
                    if decode_id_list[0] != self.word2index['END_TOKEN'] else [self.word2index['END_TOKEN']]
            decoded_words.append([self.index2word[id] for id in decode_id_list])
            # print(decoded_words)
        return decoded_words, data['loss']




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

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )
        try:
            from fairseq.fb_sequence_generator import FBSequenceGenerator
        except ModuleNotFoundError:
            pass

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            elif getattr(args, "fb_seq_gen", False):
                seq_gen_cls = FBSequenceGenerator
            else:
                seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )


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



