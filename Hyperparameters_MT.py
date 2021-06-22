import os
# import turibolt as bolt
import torch

class HP:
    def __init__(self):
        self.args = self.predefined_args()

    def predefined_args(self):
        args = {}
        args['test'] = None
        args['createDataset'] = True
        args['playDataset'] = 10
        args['reset'] = True
        args['device'] = "cuda:1" if torch.cuda.is_available() else "cpu"
        args['rootDir'] = './artifacts/'#bolt.ARTIFACT_DIR
        # args['retrain_model'] = 'LM'
        # args['retrain_model'] = 'policy'
        args['retrain_model'] = 'No'

        args['maxLength'] = 100
        args['vocabularySize'] = 40000

        args['hiddenSize'] = 100 #300
        args['numLayers'] = 1
        args['softmaxSamples'] = 0
        args['initEmbeddings'] = True
        args['embeddingSize'] = 100 #300
        # args['embeddingSource'] = "GoogleNews-vectors-negative300.bin"
        args['adaptive_softmax_cutoff'] = None
        args['decoder_learned_pos'] = False
        args["cross_self_attention"] = False
        args['decoder_normalize_before'] = False
        args['adaptive_softmax_dropout'] = 0
        args['tie_adaptive_weights'] = False
        args['eval_tokenized_bleu'] = False
        # args['adaptive_softmax_factor'] =
        # args['tie_adaptive_proj'] =
        args["quant_noise_pq"] = 0
        args["quant_noise_pq_block_size"] = 8
        args['activation_fn'] = 'relu'
        args["activation_dropout"] = 0.0
        # args["relu_dropout"] =
        args['decoder_normalize_before'] = False
        # args["char_inputs"]
        args['decoder_ffn_embed_dim'] = 1024
        args['decoder_attention_heads'] = 4
        args['attention_dropout'] = 0.0
        args["encoder_embed_dim"] = 512

        args['numEpochs'] = 1000
        args['saveEvery'] = 2000
        args['batchSize'] = 256
        args['learningRate'] = 0.001
        args['dropout'] = 0.3
        args['clip'] = 5.0

        args['encunit'] = 'lstm'
        args['decunit'] = 'lstm'
        args['enc_numlayer'] = 2
        args['dec_numlayer'] = 2

        args['maxLengthEnco'] = args['maxLength']
        args['maxLengthDeco'] = args['maxLength'] + 1

        args['temperature'] =1.0

        args['LMtype'] = 'lstm' # one of ['lstm', 'asso']


        return args

args = HP().args