
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
class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.encoderSeqs1 = []
        self.encoderSeqs2 = []
        self.label = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.decoder_lens = []


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

        # Plot some stats:
        self._printStats()

        # if args['playDataset']:
        #     self.playDataset()

    def _printStats(self):
        print('Loaded {}: {} words, {} QA'.format('LMbenchmark', len(self.word2index), len(self.trainingSamples)))


    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.datasets['train'])

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

        maxlen_def = args['maxLengthEnco'] if setname == 'train' else 511

        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            sample, raw_sentence = samples[i]

            if len(sample) > maxlen_def:
                sample = sample[:maxlen_def]

            batch.decoderSeqs.append([self.word2index['START_TOKEN']] + sample)  # Add the <go> and <eos> tokens
            batch.targetSeqs.append(sample + [self.word2index['END_TOKEN']])  # Same as decoder, but shifted to the left (ignore the <go>)

            assert len(batch.decoderSeqs[i]) <= maxlen_def +1

            # TODO: Should use tf batch function to automatically add padding and batch samples
            # Add padding & define weight
            batch.decoder_lens.append(len(batch.targetSeqs[i]))

        maxlen_dec = max(batch.decoder_lens)


        for i in range(batchSize):
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.word2index['PAD']] * (maxlen_dec - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i]  = batch.targetSeqs[i]  + [self.word2index['PAD']] * (maxlen_dec - len(batch.targetSeqs[i]))

        pre_sort_list = [(a, b, c) for a, b, c  in
                         zip( batch.decoderSeqs, batch.decoder_lens,
                             batch.targetSeqs)]

        post_sorted_list = sorted(pre_sort_list, key=lambda x: x[1], reverse=True)

        batch.decoderSeqs = [a[0] for a in post_sorted_list]
        batch.decoder_lens = [a[1] for a in post_sorted_list]
        batch.targetSeqs = [a[2] for a in post_sorted_list]

        return batch

    def getBatches(self, setname = 'train'):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        self.shuffle()


        batches = []
        batch_size = args['batchSize'] if setname == 'train' else 32
        print(len(self.datasets[setname]), setname, batch_size)
        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, self.getSampleSize(setname), batch_size):
                yield self.datasets[setname][i:min(i + batch_size, self.getSampleSize(setname))]

        # TODO: Should replace that by generator (better: by tf.queue)

        for index, samples in enumerate(genNextSamples()):
            # print([self.index2word[id] for id in samples[5][0]], samples[5][2])
            batch = self._createBatch(samples, setname)
            batches.append(batch)

        # print([self.index2word[id] for id in batches[2].encoderSeqs[5]], batches[2].raws[5])
        return batches

    def getSampleSize(self, setname = 'train'):
        """Return the size of the dataset
        Return:
            int: Number of training samples
        """
        return len(self.datasets[setname])

    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2index)

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

            self.corpusDir_train = {'EN_DE.en': self.basedir + 'europarl-v7.de-en.en',
                                    'EN_DE.de': self.basedir + 'europarl-v7.de-en.de',
                                    'EN_FR.en': self.basedir + 'europarl-v7.fr-en.en',
                                    'EN_FR.fr': self.basedir + 'europarl-v7.fr-en.fr'}
            self.corpusDir_test =  {'EN_DE.en': self.basedir + 'newstest2014-deen-src.en.sgm',
                                    'EN_DE.de': self.basedir + 'newstest2014-deen-ref.de.sgm',
                                    'EN_FR.en': self.basedir + 'newstest2014-fren-src.en.sgm',
                                    'EN_FR.fr': self.basedir + 'newstest2014-fren-ref.fr.sgm'}
            if not os.path.exists(self.basedir):
                os.mkdir(self.basedir)

            if not os.path.exists(self.corpusDir_train['En_DE.en']):
                r = requests.get('http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz', allow_redirects=True)
                open(self.basedir + 'train.tgz', 'wb').write(r.content)
                self.extract(self.basedir + 'train.tgz', self.basedir)
                r = requests.get('http://www.statmt.org/wmt14/test-full.tgz', allow_redirects=True)
                open(self.basedir + 'test.tgz', 'wb').write(r.content)
                self.extract(self.basedir + 'test.tgz', self.basedir)

            if not args['createDataset']:
                self.basedir = args['rootDir']

            self.fullSamplesPath = args['rootDir'] + '/LMdata.pkl'  # Full sentences length/vocab



        print(self.fullSamplesPath)
        datasetExist = os.path.isfile(self.fullSamplesPath)
        if not datasetExist:  # First time we load the database: creating all files
            print('Training data not found. Creating dataset...')

            dataset = {'EN_DE': {'train': [], 'valid':[], 'test':[]},
                       'EN_FR': {'train': [], 'valid':[], 'test':[]}}

            learn_bpe([self.corpusDir_train['EN_DE.en'], self.corpusDir_train['EN_DE.de']], args['rootDir'] + 'EN_DE.bpe', 37000, 6, True)
            for type in ['EN_DE', 'EN_FR']:
                total_words = []
                codes = codecs.open(args['rootDir'] + type +'.bpe', encoding='utf-8')
                bpe = BPE(codes, separator='@@')
                with open(self.corpusDir_train[type + '.en'], 'r') as src_handle:
                    with open(self.corpusDir_train[type + '.' + type[-2:].lower()], 'r') as tgt_handle:
                        src_lines = src_handle.readlines()
                        tgt_lines = tgt_handle.readlines()
                        for src_line, tgt_line in zip(src_lines, tgt_lines):
                            if len(src_line) < 5 or len(tgt_line) < 5:
                                continue
                            src_line = src_line.lower().strip()
                            tgt_line = tgt_line.lower().strip()
                            src_bpe_line = bpe.process_line(src_line)
                            tgt_bpe_line = bpe.process_line(tgt_line)
                            total_words.extend(src_bpe_line)
                            total_words.extend(tgt_bpe_line)
                            dataset[type]['train'].extend([src_bpe_line, tgt_bpe_line])

                with open(self.corpusDir_test[type + '.en'], 'r') as src_handle:
                    with open(self.corpusDir_test[type + '.' + type[-2:].lower()], 'r') as tgt_handle:
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
                                src_bpe_line = bpe.process_line(src_sen)
                                tgt_bpe_line = bpe.process_line(tgt_sen)
                                dataset[type]['test'].extend([src_bpe_line, tgt_bpe_line])



                print(type, len(dataset[type]['train']), len(dataset[type]['valid']),len(dataset[type]['test']))

                fdist = nltk.FreqDist(total_words)
                sort_count = fdist.most_common(37000)
            print('sort_count: ', len(sort_count))

            with open(self.basedir + "/voc.txt", "w") as v:
                for w, c in tqdm(sort_count):
                    if w not in [' ', '', '\n']:
                        v.write(w)
                        v.write(' ')
                        v.write(str(c))
                        v.write('\n')

                v.close()

            self.word2index = self.read_word2vec(self.basedir + '/voc.txt')
            self.sorted_word_index = sorted(self.word2index.items(), key=lambda item: item[1])
            print('sorted')
            self.index2word = [w for w, n in self.sorted_word_index]
            print('index2word')

            # with open(os.path.join(self.basedir + '/dump2.pkl'), 'wb') as handle:
            #     data = {
            #         'word2index': self.word2index,
            #         'index2word': self.index2word,
            #         'datasets': datasets
            #     }
            #     pickle.dump(data, handle, -1)


            self.index2word_set = set(self.index2word)
            print('set')

            # self.raw_sentences = copy.deepcopy(dataset)
            for setname in ['train', 'valid', 'test']:
                dataset[setname] = [(self.TurnWordID(sen), sen) for sen in tqdm(dataset[setname])]
            self.datasets = dataset


            # Saving
            print('Saving dataset...')
            self.saveDataset(self.fullSamplesPath)  # Saving tf samples
        else:
            self.loadDataset(self.fullSamplesPath)
            print('loaded')
            # print(max([len(sen) for sid, sen in self.datasets['train']]))
            # self.symbols = [str(ind) for ind, w in enumerate(self.index2word) if not w.isalpha()]
            # with open(args['rootDir'] + '/symbol_index.txt','w') as handle:
            #     handle.write(' '.join(self.symbols))
            # self.saveDataset(self.fullSamplesPath + 're')
            # print('resaved')



    def saveDataset(self, filename):
        """Save samples to file
        Args:
            filename (str): pickle filename
        """
        with open(os.path.join(filename), 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                'word2index': self.word2index,
                'index2word': self.index2word,
                'datasets': self.datasets
            }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadDataset(self, filename):
        """Load samples from file
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2index = data['word2index']
            self.index2word = data['index2word']
            self.datasets = data['datasets']


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

    def TurnWordID(self, words):
        res = []
        for w in words:
            w = w.lower()
            if w in self.index2word_set:
                id = self.word2index[w]
                # if id > 20000:
                #     print('id>20000:', w,id)
                res.append(id)
            else:
                res.append(self.word2index['UNK'])
        return res


    def printBatch(self, batch):
        """Print a complete batch, useful for debugging
        Args:
            batch (Batch): a batch object
        """
        print('----- Print batch -----')
        for i in range(len(batch.encoderSeqs[0])):  # Batch size
            print('Encoder: {}'.format(self.batchSeq2str(batch.encoderSeqs, seqId=i)))
            print('Decoder: {}'.format(self.batchSeq2str(batch.decoderSeqs, seqId=i)))
            print('Targets: {}'.format(self.batchSeq2str(batch.targetSeqs, seqId=i)))
            print('Weights: {}'.format(' '.join([str(weight) for weight in [batchWeight[i] for batchWeight in batch.weights]])))

    def sequence2str(self, sequence, clean=False, reverse=False):
        """Convert a list of integer into a human readable string
        Args:
            sequence (list<int>): the sentence to print
            clean (Bool): if set, remove the <go>, <pad> and <eos> tokens
            reverse (Bool): for the input, option to restore the standard order
        Return:
            str: the sentence
        """

        if not sequence:
            return ''

        if not clean:
            return ' '.join([self.index2word[idx] for idx in sequence])

        sentence = []
        for wordId in sequence:
            if wordId == self.word2index['END_TOKEN']:  # End of generated sentence
                break
            elif wordId != self.word2index['PAD'] and wordId != self.word2index['START_TOKEN']:
                sentence.append(self.index2word[wordId])

        if reverse:  # Reverse means input so no <eos> (otherwise pb with previous early stop)
            sentence.reverse()

        return self.detokenize(sentence)

    def detokenize(self, tokens):
        """Slightly cleaner version of joining with spaces.
        Args:
            tokens (list<string>): the sentence to print
        Return:
            str: the sentence
        """
        return ''.join([
            ' ' + t if not t.startswith('\'') and
                       t not in string.punctuation
                    else t
            for t in tokens]).strip().capitalize()

    def batchSeq2str(self, batchSeq, seqId=0, **kwargs):
        """Convert a list of integer into a human readable string.
        The difference between the previous function is that on a batch object, the values have been reorganized as
        batch instead of sentence.
        Args:
            batchSeq (list<list<int>>): the sentence(s) to print
            seqId (int): the position of the sequence inside the batch
            kwargs: the formatting options( See sequence2str() )
        Return:
            str: the sentence
        """
        sequence = []
        for i in range(len(batchSeq)):  # Sequence length
            sequence.append(batchSeq[i][seqId])
        return self.sequence2str(sequence, **kwargs)

    def sentence2enco(self, sentence):
        """Encode a sequence and return a batch as an input for the model
        Return:
            Batch: a batch object containing the sentence, or none if something went wrong
        """

        if sentence == '':
            return None

        # First step: Divide the sentence in token
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) > args['maxLength']:
            return None

        # Second step: Convert the token in word ids
        wordIds = []
        for token in tokens:
            wordIds.append(self.getWordId(token, create=False))  # Create the vocabulary and the training sentences

        # Third step: creating the batch (add padding, reverse)
        batch = self._createBatch([[wordIds, []]])  # Mono batch, no target output

        return batch

    def deco2sentence(self, decoderOutputs):
        """Decode the output of the decoder and return a human friendly sentence
        decoderOutputs (list<np.array>):
        """
        sequence = []

        # Choose the words with the highest prediction score
        for out in decoderOutputs:
            sequence.append(np.argmax(out))  # Adding each predicted word ids

        return sequence  # We return the raw sentence. Let the caller do some cleaning eventually

    def playDataset(self):
        """Print a random dialogue from the dataset
        """
        print('Randomly play samples:')
        print(len(self.datasets['train']))
        for i in range(args['playDataset']):
            idSample = random.randint(0, len(self.datasets['train']) - 1)
            print('sen: {} {}'.format(self.sequence2str(self.datasets['train'][idSample][0], clean=True), self.datasets['train'][idSample][1]))
            print()
        pass


def tqdm_wrap(iterable, *args, **kwargs):
    """Forward an iterable eventually wrapped around a tqdm decorator
    The iterable is only wrapped if the iterable contains enough elements
    Args:
        iterable (list): An iterable object which define the __len__ method
        *args, **kwargs: the tqdm parameters
    Return:
        iter: The iterable eventually decorated
    """
    if len(iterable) > 100:
        return tqdm(iterable, *args, **kwargs)
    return iterable
