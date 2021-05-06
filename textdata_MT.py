
import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random
import string, copy
from nltk.tokenize import word_tokenize
from Hyperparameters import args
import requests, tarfile
from learn_bpe import learn_bpe
from apply_bpe import BPE
from bs4 import BeautifulSoup
import codecs
class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.encoderSeqs = []
        self.label = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.encoder_lens = []
        self.decoder_lens = []
        self.raw_source = []
        self.raw_target = []


class TextData_MT:
    """Dataset class
    Warning: No vocabulary limit
    """


    def __init__(self, corpusname):
        """Load all conversations
        Args:
            args: parameters of the model
        """

        # Path variables
        self.tokenizer = word_tokenize


        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]

        self.word2index = {}
        self.index2word = {}  # For a rapid conversion (Warning: If replace dict by list, modify the filtering to avoid linear complexity with del)

        self.loadCorpus(corpusname)


    def _printStats(self):
        print('Loaded {}: {} words, {} QA'.format('LMbenchmark', len(self.word2index), len(self.trainingSamples)))


    def shuffle(self, typename):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.datasets[typename]['train'])

    def _createBatch(self, samples, setname):
        """Create a single batch from the list of sample. The batch size is automatically defined by the number of
        samples given.
        The inputs should already be inverted. The target should already have <go> and <eos>
        Warning: This function should not make direct calls to args['batchSize'] !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """

        batch = Batch()
        batchSize = len(samples)

        maxlen_def = args['maxLengthEnco'] #if setname == 'train' else 511

        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            src_sample, tgt_sample, raw_src, raw_tgt = samples[i]

            if len(src_sample) > maxlen_def:
                src_sample = src_sample[:maxlen_def]
            if len(tgt_sample) > maxlen_def:
                tgt_sample = tgt_sample[:maxlen_def]

            batch.encoderSeqs.append(src_sample)
            batch.decoderSeqs.append([self.word2index[args['typename']]['START_TOKEN']] + src_sample)  # Add the <go> and <eos> tokens
            batch.targetSeqs.append(src_sample + [self.word2index[args['typename']]['END_TOKEN']])  # Same as decoder, but shifted to the left (ignore the <go>)

            assert len(batch.decoderSeqs[i]) <= maxlen_def +1

            # TODO: Should use tf batch function to automatically add padding and batch samples
            # Add padding & define weight
            batch.encoder_lens.append(len(batch.encoderSeqs[i]))
            batch.decoder_lens.append(len(batch.targetSeqs[i]))
            batch.raw_source.append(raw_src)
            batch.raw_target.append(raw_tgt)

        maxlen_dec = max(batch.decoder_lens)
        maxlen_enc = max(batch.encoder_lens)


        for i in range(batchSize):
            batch.encoderSeqs[i] = batch.encoderSeqs[i] + [self.word2index[args['typename']]['PAD']] * (maxlen_enc - len(batch.encoderSeqs[i]))
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.word2index[args['typename']]['PAD']] * (maxlen_dec - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i]  = batch.targetSeqs[i]  + [self.word2index[args['typename']]['PAD']] * (maxlen_dec - len(batch.targetSeqs[i]))

        # pre_sort_list = [(a, b, c) for a, b, c  in
        #                  zip( batch.decoderSeqs, batch.decoder_lens,
        #                      batch.targetSeqs)]
        #
        # post_sorted_list = sorted(pre_sort_list, key=lambda x: x[1], reverse=True)
        #
        # batch.decoderSeqs = [a[0] for a in post_sorted_list]
        # batch.decoder_lens = [a[1] for a in post_sorted_list]
        # batch.targetSeqs = [a[2] for a in post_sorted_list]

        return batch

    def getBatches(self, typename, setname = 'train'):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        self.shuffle(typename)


        batches = []
        batch_size = args['batchSize'] #if setname == 'train' else 32
        print(len(self.datasets[typename][setname]), typename, setname, batch_size)
        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, self.getSampleSize(typename, setname), batch_size):
                yield self.datasets[typename][setname][i:min(i + batch_size, self.getSampleSize(typename, setname))]

        # TODO: Should replace that by generator (better: by tf.queue)

        for index, samples in enumerate(genNextSamples()):
            # print([self.index2word[id] for id in samples[5][0]], samples[5][2])
            batch = self._createBatch(samples, setname)
            batches.append(batch)

        # print([self.index2word[id] for id in batches[2].encoderSeqs[5]], batches[2].raws[5])
        return batches

    def getSampleSize(self, typename, setname = 'train'):
        """Return the size of the dataset
        Return:
            int: Number of training samples
        """
        return len(self.datasets[typename][setname])

    def getVocabularySize(self, typename):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2index[typename])

    def extract(self, tar_url, extract_path='.'):
        print(tar_url)
        tar = tarfile.open(tar_url, 'r')
        for item in tar:
            tar.extract(item, extract_path)

    def loadCorpus(self, corpusname):
        """Load/create the conversations data
        """
        # self.corpusDir = '../Gutenberg_data/' + corpusname + '/' if args['createDataset'] else args['rootDir']
        if corpusname == 'MT':
            if args['server'] == 'dgx':
                self.basedir = './.data/MT/'
            else:
                self.basedir = '../MT/'

            # self.corpusDir_train = {'EN_DE.en': self.basedir + 'training/europarl-v7.de-en.en',
            #                         'EN_DE.de': self.basedir + 'training/europarl-v7.de-en.de',
            #                         'EN_FR.en': self.basedir + 'training/europarl-v7.fr-en.en',
            #                         'EN_FR.fr': self.basedir + 'training/europarl-v7.fr-en.fr'}
            self.corpusDir_train = {'EN_DE.en': self.basedir + 'training/news-commentary-v9.de-en.en',
                                    'EN_DE.de': self.basedir + 'training/news-commentary-v9.de-en.de',
                                    'EN_FR.en': self.basedir + 'training/news-commentary-v9.fr-en.en',
                                    'EN_FR.fr': self.basedir + 'training/news-commentary-v9.fr-en.fr'}
            self.corpusDir_dev =  {'EN_DE.en': self.basedir + 'dev/news-test2008-src.en.sgm',
                                    'EN_DE.de': self.basedir + 'dev/news-test2008-ref.de.sgm',
                                    'EN_FR.en': self.basedir + 'dev/news-test2008-src.en.sgm',
                                    'EN_FR.fr': self.basedir + 'dev/news-test2008-ref.fr.sgm'}
            self.corpusDir_test =  {'EN_DE.en': self.basedir + 'test-full/newstest2014-deen-src.en.sgm',
                                    'EN_DE.de': self.basedir + 'test-full/newstest2014-deen-ref.de.sgm',
                                    'EN_FR.en': self.basedir + 'test-full/newstest2014-fren-src.en.sgm',
                                    'EN_FR.fr': self.basedir + 'test-full/newstest2014-fren-ref.fr.sgm'}
            if not os.path.exists(self.basedir):
                os.mkdir(self.basedir)

            if not os.path.exists(self.corpusDir_train['EN_DE.en']):
                # os.system('wget -P ' + self.basedir + ' http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz')
                # os.system('tar zxvf training-parallel-europarl-v7.tgz')
                r = requests.get('http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz', allow_redirects=True)
                open(self.basedir + 'train.tgz', 'wb').write(r.content)
                self.extract(self.basedir + 'train.tgz', self.basedir)

                # os.system('wget -P ' + self.basedir + ' http://www.statmt.org/wmt14/dev.tgz')
                # os.system('tar zxvf dev.tgz')
                r = requests.get('http://www.statmt.org/wmt14/dev.tgz', allow_redirects=True)
                open(self.basedir + 'dev.tgz', 'wb').write(r.content)
                self.extract(self.basedir + 'dev.tgz', self.basedir)
                # os.system('wget -P ' + self.basedir + ' http://www.statmt.org/wmt14/test-full.tgz')
                # os.system('tar zxvf test-full.tgz')
                r = requests.get('http://www.statmt.org/wmt14/test-full.tgz', allow_redirects=True)
                open(self.basedir + 'test.tgz', 'wb').write(r.content)
                self.extract(self.basedir + 'test.tgz', self.basedir)

            if not args['createDataset']:
                self.basedir = args['rootDir']

            self.fullSamplesPath = args['rootDir'] + '/MT2.pkl'  # Full sentences length/vocab

        print(self.fullSamplesPath)
        datasetExist = os.path.isfile(self.fullSamplesPath)
        if not datasetExist:  # First time we load the database: creating all files
            print('Training data not found. Creating dataset...')

            dataset = {'EN_DE': {'train': [], 'valid':[], 'test':[]},
                       'EN_FR': {'train': [], 'valid':[], 'test':[]}}
            self.word2index = {'EN_DE': {}, 'EN_FR':{}}
            self.sorted_word_index = {'EN_DE': {}, 'EN_FR':{}}
            self.index2word = {'EN_DE': {}, 'EN_FR':{}}
            self.index2word_set = {'EN_DE': {}, 'EN_FR':{}}

            learn_bpe([self.corpusDir_train['EN_DE.en'], self.corpusDir_train['EN_DE.de']], args['rootDir'] + 'EN_DE.bpe', 37000, 6, True)
            learn_bpe([self.corpusDir_train['EN_FR.en'], self.corpusDir_train['EN_FR.fr']], args['rootDir'] + 'EN_FR.bpe', 37000, 6, True)
            for typename in ['EN_DE', 'EN_FR']:
                total_words = []
                codes = codecs.open(args['rootDir'] + typename +'.bpe', encoding='utf-8')
                bpe = BPE(codes, separator='@@')
                with open(self.corpusDir_train[typename + '.en'], 'r') as src_handle:
                    with open(self.corpusDir_train[typename + '.' + typename[-2:].lower()], 'r') as tgt_handle:
                        src_lines = src_handle.readlines()
                        tgt_lines = tgt_handle.readlines()
                        for src_line, tgt_line in zip(src_lines, tgt_lines):
                            if len(src_line) < 5 or len(tgt_line) < 5:
                                continue
                            src_line = src_line.lower().strip()
                            tgt_line = tgt_line.lower().strip()
                            src_bpe_line = bpe.process_line(src_line).split()
                            tgt_bpe_line = bpe.process_line(tgt_line).split()
                            total_words.extend(src_bpe_line)
                            total_words.extend(tgt_bpe_line)
                            dataset[typename]['train'].append([src_bpe_line, tgt_bpe_line])

                with open(self.corpusDir_dev[typename + '.en'], 'r') as src_handle:
                    with open(self.corpusDir_dev[typename + '.' + typename[-2:].lower()], 'r') as tgt_handle:
                        src_content = src_handle.read()
                        tgt_content = tgt_handle.read()
                        src_soup = BeautifulSoup(src_content)
                        tgt_soup = BeautifulSoup(tgt_content)
                        src_docs = src_soup.find_all('doc')
                        tgt_docs = tgt_soup.find_all('doc')
                        assert len(src_docs) == len(tgt_docs)

                        for srcd, tgtd in zip(src_docs, tgt_docs):
                            assert srcd.attrs['docid'] == tgtd.attrs['docid']
                            src_segs = srcd.find_all('seg')
                            tgt_segs = tgtd.find_all('seg')
                            for srcseg, tgtseg in zip(src_segs, tgt_segs):
                                src_sen = srcseg.text
                                tgt_sen = tgtseg.text
                                src_bpe_line = bpe.process_line(src_sen).split()
                                tgt_bpe_line = bpe.process_line(tgt_sen).split()
                                dataset[typename]['valid'].append([src_bpe_line, tgt_bpe_line])

                with open(self.corpusDir_test[typename + '.en'], 'r') as src_handle:
                    with open(self.corpusDir_test[typename + '.' + typename[-2:].lower()], 'r') as tgt_handle:
                        src_content = src_handle.read()
                        tgt_content = tgt_handle.read()
                        src_soup = BeautifulSoup(src_content)
                        tgt_soup = BeautifulSoup(tgt_content)
                        src_docs = src_soup.find_all('doc')
                        tgt_docs = tgt_soup.find_all('doc')
                        assert len(src_docs) == len(tgt_docs)

                        for srcd, tgtd in zip(src_docs, tgt_docs):
                            assert srcd.attrs['docid'] == tgtd.attrs['docid']
                            src_segs = srcd.find_all('seg')
                            tgt_segs = tgtd.find_all('seg')
                            for srcseg, tgtseg in zip(src_segs, tgt_segs):
                                src_sen = srcseg.text
                                tgt_sen = tgtseg.text
                                src_bpe_line = bpe.process_line(src_sen).split()
                                tgt_bpe_line = bpe.process_line(tgt_sen).split()
                                dataset[typename]['test'].append([src_bpe_line, tgt_bpe_line])

                print(typename, len(dataset[typename]['train']), len(dataset[typename]['valid']),len(dataset[typename]['test']))

                fdist = nltk.FreqDist(total_words)
                sort_count = fdist.most_common(37000)
                print('sort_count: ', len(sort_count))

                with open(args['rootDir'] + "/" +typename + "_voc.txt", "w") as v:
                    for w, c in tqdm(sort_count):
                        if w not in [' ', '', '\n']:
                            v.write(w)
                            v.write(' ')
                            v.write(str(c))
                            v.write('\n')

                    v.close()

                self.word2index[typename] = self.read_word2vec(args['rootDir'] + "/" +typename + '_voc.txt')
                self.sorted_word_index[typename] = sorted(self.word2index[typename].items(), key=lambda item: item[1])
                print('sorted')
                self.index2word[typename] = [w for w, n in self.sorted_word_index[typename]]
                print('index2word')
                self.index2word_set[typename] = set(self.index2word[typename])
                print('set')

            # self.raw_sentences = copy.deepcopy(dataset)

                for setname in ['train', 'valid', 'test']:
                    dataset[typename][setname] = [(self.TurnWordID(src, typename), self.TurnWordID(tgt, typename), src, tgt) for src,tgt in tqdm(dataset[typename][setname])]
            self.datasets = dataset


            # Saving
            print('Saving dataset...')
            self.saveDataset(self.fullSamplesPath)  # Saving tf samples
        else:
            self.loadDataset(self.fullSamplesPath)

            print('loaded')
            for typename in ['EN_DE', 'EN_FR']:
                print(typename + ' Vocab size: ', len(self.word2index[typename]))
                for setname in ['train', 'valid', 'test']:
                    print(typename + ' ' + setname + ' size: ', len(self.datasets[typename][setname]))



    def saveDataset(self, filename):
        """Save samples to file
        Args:
            filename (str): pickle filename
        """
        with open(os.path.join(filename), 'wb') as handle:
            data = {'EN_DE': {  # Warning: If adding something here, also modifying loadDataset
                    'word2index': self.word2index['EN_DE'],
                    'index2word': self.index2word['EN_DE'],
                    'datasets': self.datasets['EN_DE']
                },
                'EN_FR': {  # Warning: If adding something here, also modifying loadDataset
                    'word2index': self.word2index['EN_FR'],
                    'index2word': self.index2word['EN_FR'],
                    'datasets': self.datasets['EN_FR']
                }
            }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadDataset(self, filename):
        """Load samples from file
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))

        self.word2index= {}
        self.index2word= {}
        self.datasets = {}
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2index['EN_DE'] = data['EN_DE']['word2index']
            self.index2word['EN_DE'] = data['EN_DE']['index2word']
            self.datasets['EN_DE'] = data['EN_DE']['datasets']
            self.word2index['EN_FR'] = data['EN_FR']['word2index']
            self.index2word['EN_FR'] = data['EN_FR']['index2word']
            self.datasets['EN_FR'] = data['EN_FR']['datasets']


    def read_word2vec(self, vocfile ):
        word2index = dict()
        word2index['PAD'] = 0
        word2index['START_TOKEN'] = 1
        word2index['END_TOKEN'] = 2
        word2index['UNK'] = 3
        cnt = 4
        with open(vocfile, "r") as v:

            for line in v:
                word = line.strip().split()[0]
                word2index[word] = cnt
                print(word,cnt)
                cnt += 1

        print(len(word2index),cnt)
        # dic = {w:numpy.random.normal(size=[int(sys.argv[1])]).astype('float32') for w in word2index}
        print ('Dictionary Got!')
        return word2index

    def TurnWordID(self, words, type):
        res = []
        for w in words:
            w = w.lower()
            if w in self.index2word_set[type]:
                id = self.word2index[type][w]
                # if id > 20000:
                #     print('id>20000:', w,id)
                res.append(id)
            else:
                res.append(self.word2index[type]['UNK'])
        return res
