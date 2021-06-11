# Copyright 2020 . All Rights Reserved.
# Author : Lei Sha
import functools
print = functools.partial(print, flush=True)
from TranslationModel import TranslationModel

from textdata_MT_preprocessed import  TextData_MT
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
from Hyperparameters_MT import args
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# import matplotlib.pyplot as plt
import numpy as np
import copy
# from inverse_square_root_schedule import InverseSquareRootSchedule, InverseSquareRootLRScheduleConfig
import fairseq
from fairseq.optim import lr_scheduler
import utils

from fairseq import (
    tasks,
)

# from kenLM import LMEvaluator as LMEr
from omegaconf import dictconfig
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
    args['typename'] = args['corpus']

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

        self.l1, self.l2 = args['typename'].split('_')

    def change_word(self, d):

        d.indices['START_TOKEN']=0
        d.indices['PAD']=1
        d.indices['END_TOKEN']=2
        d.indices['UNK']=3
        d.symbols[0]='START_TOKEN'
        d.symbols[1]='PAD'
        d.symbols[2] = 'END_TOKEN'
        d.symbols[3] ='UNK'
        d.bos_word = 'START_TOKEN'
        d.eos_word = 'END_TOKEN'
        d.pad_word = 'PAD'
        d.unk_word = 'UNK'

    def main(self):
        self.textData =  TextData_MT('MT')

        args['maxLengthEnco'] = args['maxLength'] = 200
        args['maxLengthDeco'] =  args['maxLengthEnco'] + 1
        self.start_token = self.textData.word2index[args['typename']][self.l2]['START_TOKEN']
        self.end_token = self.textData.word2index[args['typename']][self.l2]['END_TOKEN']
        torch.manual_seed(0)

        self.task = tasks.setup_task(dictconfig.DictConfig(
            {'_name': 'translation', 'data': 'data-bin/iwslt14.tokenized.de-en', 'source_lang': None,
             'target_lang': None, 'load_alignments': False, 'left_pad_source': True, 'left_pad_target': False,
             'max_source_positions': 1024, 'max_target_positions': 1024, 'upsample_primary': -1,
             'truncate_source': False, 'num_batch_buckets': 0, 'train_subset': 'train', 'dataset_impl': None,
             'required_seq_len_multiple': 1, 'eval_bleu': True,
             'eval_bleu_args': '{"beam":5,"max_len_a":1.2,"max_len_b":10}', 'eval_bleu_detok': 'moses',
             'eval_bleu_detok_args': '{}', 'eval_tokenized_bleu': False, 'eval_bleu_remove_bpe': '@@ ',
             'eval_bleu_print_samples': True}))

        self.change_word(self.task.src_dict)
        self.change_word(self.task.tgt_dict)
        self.textData.word2index[args['typename']][self.l1] = self.task.src_dict.indices
        self.textData.index2word[args['typename']][self.l1] =  self.task.src_dict.symbols
        self.textData.word2index[args['typename']][self.l2]= self.task.tgt_dict.indices
        self.textData.index2word[args['typename']][self.l2] =  self.task.tgt_dict.symbols
        args['vocabularySize_src'] = self.textData.getVocabularySize(args['typename'], self.l1)
        args['vocabularySize_tgt'] = self.textData.getVocabularySize(args['typename'], self.l2)

        print(args['vocabularySize_src'], args['vocabularySize_tgt'])
        print(args)
        self.task.load_dataset('valid', combine=False, epoch=1)
        self.train_itr = self.get_train_iterator(
            epoch=1, load_dataset=True
        )

        self.model = TranslationModel(self.textData.word2index[args['typename']][self.l1],
                                      self.textData.index2word[args['typename']][self.l1],
                                      self.textData.word2index[args['typename']][self.l2],
                                      self.textData.index2word[args['typename']][self.l2]).to(args['device'])

        # self.model = torch.load(self.model_path.replace('model', 'model_'+'fw'), map_location=args['device'])
        params = sum([np.prod(p.size()) for p in self.model.parameters()])
        print(params, sum([sys.getsizeof(p.storage()) for p in self.model.parameters()])/1000000, 'm')
        self.trainMT()     # contain  model saving
        self.evaluateRandomly()
        self.testMT()

    def lr_step_begin_epoch(self, epoch, lr_scheduler):
        """Adjust the learning rate at the beginning of the epoch."""
        lr_scheduler.step_begin_epoch(epoch)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update(lr_scheduler)

    def lr_step(self, epoch, lr_scheduler, val_loss=None):
        """Adjust the learning rate at the end of the epoch."""
        # print(val_loss)
        lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update(lr_scheduler)

    def lr_step_update(self, lr_scheduler):
        """Update the learning rate after each update."""
        new_lr = lr_scheduler.step_update(self.get_num_updates())
        # if isinstance(new_lr, dict):
        #     for k, v in new_lr.items():
        #         metrics.log_scalar(f"lr_{k}", v, weight=0, priority=300)
        #     new_lr = new_lr.get("default", next(iter(new_lr.values())))
        # else:
        #     metrics.log_scalar("lr", new_lr, weight=0, priority=300)
        return new_lr

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        # self.lr_step_update()
        # if self.quantizer:
        #     self.quantizer.step_update(self._num_updates)
        # metrics.log_scalar("num_updates", self._num_updates, weight=0, priority=200)

    def trainMT(self, print_every=10000, plot_every=10, learning_rate=0.001):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        print(type(self.textData.word2index[args['typename']]), args['device'])

        # learning_rate =
        # optimizer = optim.Adam(self.model.parameters(), lr=0.0005,betas=(0.9, 0.98), eps=1e-08, weight_decay = 0.0001)#, amsgrad=True)
        optimizer = fairseq.optim.build_optimizer(dictconfig.DictConfig({'_name': 'adam', 'adam_betas': '(0.9,0.98)', 'adam_eps': 1e-08, 'weight_decay': 0.0001, 'use_old_adam': False, 'tpu': False, 'lr': [0.0005]}), self.model.parameters())
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=5.0)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        scheduler = lr_scheduler.build_lr_scheduler(
            dictconfig.DictConfig({'_name': 'inverse_sqrt', 'warmup_updates': 4000, 'warmup_init_lr': -1.0, 'lr': [0.0005]}),
            optimizer,
        )
        iter = 1
        batches = self.textData.getBatches(args['typename'], tgt_lang = self.l2)
        n_iters = len(batches)
        print('niters ',n_iters)

        args['trainseq2seq'] = False

        min_BLEU = -1

        self._num_updates = 0
        self.Cal_BLEU_for_dataset('test')

        CE_loss_total = 0
        KL_total = 0
        VAE_recon_total = 0
        error_total = 0
        print('begin training...')
        epoch = 1
        self.lr_step_begin_epoch(epoch,scheduler)
        while epoch < args['numEpochs']:
            losses = []
            itr = self.train_itr.next_epoch_itr(
                fix_batches_to_gpus=False ,
                shuffle=(self.train_itr.next_epoch_idx > 0),
            )

            # for batch in batches:
            for sample in itr:
                optimizer.zero_grad()
                # x={}
                # x['id'] = torch.LongTensor(batch.id).to(args['device'])
                # x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
                # x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs)).to(args['device'])
                # x['dec_len'] = batch.decoder_lens
                # x['target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs)).to(args['device'])


                loss, data = self.model(sample)    # batch seq_len outsize
                loss_mean = torch.mean(loss)
                # Reward = loss_mean.data

                loss_mean.backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args['clip'])

                optimizer.step()


                print_loss_total += loss_mean.item()
                plot_loss_total += loss_mean.item()

                losses.append(loss_mean.item())
                self.set_num_updates(self.get_num_updates() + 1)
                self.lr_step_update(scheduler)
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
                        BLEU, bleu_ori,val_losses = self.Cal_BLEU_for_dataset('test')
                        print('Test ppl: ', BLEU, bleu_ori, 'Val loss: ', sum(val_losses)/ len(val_losses))

                        if BLEU > min_BLEU or min_BLEU == -1:
                            print('BLEU = ', BLEU, bleu_ori, '>= min_BLEU (', min_BLEU, '), saving model...')
                            min_BLEU = BLEU

                if iter % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0

                iter+=1

            # scheduler.step()
            #     for g in optimizer.param_groups:
            #         g['lr'] = args['embeddingSize'] ** (-0.5) * min(iter ** (-0.5), iter*(4000**(-1.5)))
            BLEU, bleu_ori, valid_losses = self.Cal_BLEU_for_dataset('test')
            if BLEU > min_BLEU or min_BLEU == -1:
                print('BLEU = ', BLEU, bleu_ori, '>= min_BLEU (', min_BLEU, '), saving model...')
                # torch.save(self.model, self.model_path.replace('model', 'model_'+args['LMtype']+'_'+args['corpus'] + '_'+str(args['maxLength'])+'_'+str(args['numLayers'])+'_'+str(args['embeddingSize'])))
                min_BLEU = BLEU
            lr = self.lr_step(epoch, scheduler,valid_losses[0])
            print('Epoch ', epoch, 'loss = ', sum(losses) / len(losses), 'Valid BLEU = ', BLEU, bleu_ori,'best BLEU: ', min_BLEU)
            epoch += 1

    def testMT(self):
        start = time.time()
        print('Test set BLEU = ', self.Cal_BLEU_for_dataset('test'))
        end = time.time()
        print('Test time: ', time.strftime("%H:%M:%S", time.gmtime(end-start )))

    def Cal_BLEU_for_dataset(self, datasetname):
        EVAL_BLEU_ORDER = 4
        if not hasattr(self, 'testbatches' ):
            self.testbatches = {}
        if datasetname not in self.testbatches:
            self.testbatches[datasetname] = self.textData.getBatches(args['typename'], setname=datasetname, tgt_lang = self.l2)
        self.model.eval()
        num = 0
        ave_loss = 0
        pred_ans = []  # bleu
        gold_ans = []
        rec = None
        valid_loss = []
        logs = []

        valid_epoch_itr = self.get_valid_iterator('valid')
        print(type(valid_epoch_itr),dir(valid_epoch_itr))
        itr = valid_epoch_itr.next_epoch_itr (
            shuffle=False#, set_dataset_epoch=False  # use a fixed valid set
        )
        with torch.no_grad():
            # print(len(self.testbatches[datasetname][0].decoderSeqs))
            # for batch in self.testbatches[datasetname]:
            for sample in itr:

                # x = {}
                #
                # x['id'] = torch.LongTensor(batch.id).to(args['device'])
                # x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
                # x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs)).to(args['device'])
                # x['dec_len'] = batch.decoder_lens
                # x['target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs)).to(args['device'])

                loss, sample_size, logging_output = self.model.predict(sample)    # batch seq_len outsize
                logs.append(logging_output)
                pred_ans.extend(logging_output['hyps'])
                gold_ans.extend([[r] for r in logging_output['refs']])
                valid_loss.append(loss)
                # if rec is None:
                #     rec = (decoded_words[0], batch.raw_source[0], batch.raw_target[0])

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum([log["_bleu_counts_" + str(i)] for log in logs]))
                totals.append(sum([log["_bleu_totals_" + str(i)] for log in logs]))
            metrics = {}
            metrics["_bleu_counts"] = np.array(counts)
            metrics["_bleu_totals"]=np.array(totals)
            metrics["_bleu_sys_len"]=sum([log["_bleu_sys_len"] for log in logs])
            metrics["_bleu_ref_len"]=sum([log["_bleu_ref_len"] for log in logs])
            print(metrics)

            def compute_bleu(meters):
                import inspect
                import sacrebleu

                fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                if "smooth_method" in fn_sig:
                    smooth = {"smooth_method": "exp"}
                else:
                    smooth = {"smooth": "exp"}
                bleu = sacrebleu.compute_bleu(
                    correct=np.array(meters["_bleu_counts"]),
                    total=np.array(meters["_bleu_totals"]),
                    sys_len=meters["_bleu_sys_len"],
                    ref_len=meters["_bleu_ref_len"],
                    **smooth
                )
                return round(bleu.score, 2)

            metrics["bleu"] = compute_bleu(metrics)
            print(rec)
            bleu = metrics["bleu"]
            bleu_ori = corpus_bleu(gold_ans, pred_ans)

        self.model.train()
        return bleu, bleu_ori, valid_loss

    def get_train_iterator(
        self,
        epoch,
        combine=True,
        load_dataset=True,
        data_selector=None,
        shard_batch_itr=True,
        disable_iterator_cache=False,
    ):
        self.data_parallel_world_size = 1
        self.data_parallel_rank = 0
        """Return an EpochBatchIterator over the training set for a given epoch."""
        if load_dataset:
            print("loading train data for epoch {}".format(epoch))
            self.task.load_dataset(
                'train',
                epoch=epoch,
                combine=combine,
                data_selector=data_selector,
                tpu=False,
            )
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset('train'),
            max_tokens=4096,
            max_sentences=None,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                # self.model.max_positions(),
                4096,
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=8,
                seed=1,
            num_shards=self.data_parallel_world_size if shard_batch_itr else 1,
            shard_id=self.data_parallel_rank if shard_batch_itr else 0,
                num_workers=1,
            epoch=epoch,
            data_buffer_size=10,
            disable_iterator_cache=disable_iterator_cache,
        )
        # self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def get_valid_iterator(
        self,
        subset,
        disable_iterator_cache=False,
    ):
        self.data_parallel_world_size = 1
        self.data_parallel_rank = 0
        """Return an EpochBatchIterator over given validation subset for a given epoch."""
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset(subset),
            max_tokens=4096/2,
            max_sentences=None,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                # self.model.max_positions(),
            ),
            ignore_invalid_inputs=False,
            required_batch_size_multiple=8,
            seed=1,
            num_shards=self.data_parallel_world_size,
            shard_id=self.data_parallel_rank,
            num_workers=1,
            # always pass a fixed "epoch" to keep validation data consistent
            # across training epochs
            epoch=1,
            data_buffer_size=10,
            disable_iterator_cache=disable_iterator_cache,
        )
        # self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator
 

if __name__ == '__main__':
    args['corpus'] = 'DE_EN'
    args['typename'] = args['corpus']
    # args['LMtype'] = 'transformer'
    args['norm_attn'] = True
    r = Runner()
    r.main()