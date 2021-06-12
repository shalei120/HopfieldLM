import functools
print = functools.partial(print, flush=True)
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import numpy as np

import datetime, json
from Hyperparameters_MT import args
from queue import PriorityQueue
import copy, utils
from argparse import Namespace

# from kenLM import LMEvaluator as LMEr

from modules import Hopfield, HopfieldPooling, HopfieldLayer
from modules.transformer import  HopfieldEncoderLayer
from EnergyTransformer import EnergyTransformerEncoderLayer,EnergyTransformerEncoder
from Transformer_for_MT import TransformerModel
from sequence_generator import SequenceGenerator
import search, data_utils
class TranslationModel(nn.Module):
    def __init__(self, enc_w2i, enc_i2w, dec_w2i, dec_i2w):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(TranslationModel, self).__init__()
        print("TranslationModel creation...")

        self.enc_word2index = enc_w2i
        self.enc_index2word = enc_i2w
        self.dec_word2index = dec_w2i
        self.dec_index2word = dec_i2w
        self.max_length = args['maxLengthDeco']
        self.batch_size = args['batchSize']
        self.ignore_prefix_size = 0
        self.padding_idx = enc_w2i['PAD']
        self.dtype = 'float32'

        # self.embedding_src = nn.Embedding(args['vocabularySize_src'], args['embeddingSize']).to(args['device'])
        # self.embedding_tgt = nn.Embedding(args['vocabularySize_tgt'], args['embeddingSize']).to(args['device'])

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

        self.trans_net = TransformerModel(self.enc_word2index, self.enc_index2word, self.dec_word2index, self.dec_index2word)

        gen_args = json.loads('{"beam":5,"max_len_a":1.2,"max_len_b":10}')
        self.sequence_generator = self.build_generator(
            [self.trans_net], Namespace(**gen_args)
        )

    # def generate_square_subsequent_mask(self, sz):
    #     # mask = torch.logical_not(torch.triu(torch.ones(sz, sz)) == 1)
    #     # mask[0, 0] = True
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = self.trans_net.decoder.get_normalized_probs(net_output, log_probs=True)
        target = sample["target"]
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def label_smoothed_nll_loss(self, lprobs, target, epsilon, ignore_index=None, reduce=True):
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = epsilon / (lprobs.size(-1) - 1)
        loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    def build(self, x, epsilon = 0.1, reduce=True):

        # batch_size = self.decoderInputs.size()[0]
        # self.dec_len = self.decoderInputs.size()[1]
        # enc_input_embed = self.embedding_src(self.encoderInputs)
        # dec_input_embed = self.embedding_tgt(self.decoderInputs)
        # mask = torch.sign(self.decoderTargets.float())

        net_output = self.trans_net(x)
        lprobs, target = self.get_lprobs_and_target(self.trans_net, net_output, x)
        loss, nll_loss = self.label_smoothed_nll_loss(
            lprobs,
            target,
            epsilon,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        sample_size = (
            x["target"].size(0)
        )
        logging_output = {
            "loss": loss.data / x['ntokens'] / np.log(2),
            "nll_loss": nll_loss.data,
            "ntokens": x['ntokens'],
            "nsentences": x["target"].size(0),
            "sample_size": sample_size,
        }
        loss = loss / x['ntokens'] / np.log(2)
        # attn_list = net_output[1]['attn_list']
        # if self.report_accuracy:
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        logging_output["n_correct"] = n_correct.data
        logging_output["total"] = total.data
        return loss, sample_size, logging_output


    # def forward(self, x):
    def forward(self, sample):

        sample['net_input']['src_tokens'] =  sample['net_input']['src_tokens'].to(args['device'])
        sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].to(args['device'])
        sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].to(args['device'])
        sample['target'] = sample['target'].to(args['device'])
        x={
            'enc_input':sample['net_input']['src_tokens'],
            'dec_input':sample['net_input']['prev_output_tokens'],
            'target' : sample['target']
        }
        mask = torch.sign(x['dec_input'].float())
        x['ntokens']=mask.sum()
        loss, sample_size, logging_output = self.build(x)
        return loss,logging_output

    # def predict(self, x, EVAL_BLEU_ORDER = 4):
    def predict(self, sample, EVAL_BLEU_ORDER = 4):

        sample['net_input']['src_tokens'] =  sample['net_input']['src_tokens'].to(args['device'])
        sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].to(args['device'])
        sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].to(args['device'])
        sample['target'] = sample['target'].to(args['device'])
        x={
            'enc_input':sample['net_input']['src_tokens'],
            'dec_input':sample['net_input']['prev_output_tokens'],
            'target' : sample['target']
        }

        mask = torch.sign(x['dec_input'].float())
        x['ntokens']=mask.sum()
        # sample={
        #     'id': x['id'],
        #     'nsentences':len(x['id']),
        #     'ntokens':mask.sum(),
        #     'net_input':{
        #         'src_tokens':x['enc_input'],
        #         'src_lengths':torch.sign(x['enc_input'].float()).sum(1) ,
        #         'prev_output_tokens':x['dec_input'],
        #     },
        #     'target':x['target']
        # }
        # model.eval()

        with torch.no_grad():
            loss, sample_size, logging_output  = self.build(x)
        bleu, hyps,refs = self._inference_with_bleu(self.sequence_generator, sample, self.trans_net)

        logging_output["_bleu_sys_len"] = bleu.sys_len
        logging_output["_bleu_ref_len"] = bleu.ref_len
        logging_output["hyps"] = hyps
        logging_output["refs"] = refs
        assert len(bleu.counts) == EVAL_BLEU_ORDER
        for i in range(EVAL_BLEU_ORDER):
            logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
            logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

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

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.Make_string(self.dec_word2index,
                toks.int().cpu(),
                '@@ ',
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            # if self.tokenizer:
            #     s = self.tokenizer.decode(s)
            return s

        # gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        with torch.no_grad():
            gen_out = generator.generate(
                [model], sample, prefix_tokens=None, constraints=None
            )
        hyps, refs = [], []
        # print([len(g) for g in gen_out])
        for i in range(len(gen_out)):
            endpos = gen_out[i][0]["tokens"].index('END_TOKEN')
            gen_out[i][0]["tokens"] = gen_out[i][0]["tokens"][:endpos]
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.dec_word2index['PAD']),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        # if self.cfg.eval_bleu_print_samples:
        #     logger.info("example hypothesis: " + hyps[0])
        #     logger.info("example reference: " + refs[0])

        print("example hypothesis: " + hyps[0])
        print("example reference: " + refs[0])
        res = None
        if args['eval_tokenized_bleu']:
            res = sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            res =  sacrebleu.corpus_bleu(hyps, [refs])

        return res, hyps, refs

    def Make_string(
        self,tgt_dict,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
        include_eos=False,
        separator=" ",
    ):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(
                self.Make_string(tgt_dict, t, bpe_symbol, escape_unk, extra_symbols_to_ignore, include_eos=include_eos)
                for t in tensor
            )

        extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
        extra_symbols_to_ignore.add(tgt_dict['END_TOKEN'])

        def token_string(i):
            if i == tgt_dict['UNK']:
                if unk_string is not None:
                    return unk_string
                else:
                    return tgt_dict['UNK']
            else:
                return self.dec_index2word[i]

        if 'START_TOKEN' in  tgt_dict:
            extra_symbols_to_ignore.add(tgt_dict['START_TOKEN'])

        sent = separator.join(
            token_string(i)
            for i in tensor
            if i.data not in extra_symbols_to_ignore
        )

        return data_utils.post_process(sent, bpe_symbol)

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        self.target_dictionary = self.dec_word2index

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



