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
        args['numLayers'] = 2
        args['softmaxSamples'] = 0
        args['initEmbeddings'] = True
        args['embeddingSize'] = 100 #300
        # args['embeddingSource'] = "GoogleNews-vectors-negative300.bin"

        args['numEpochs'] = 100
        args['saveEvery'] = 2000
        args['batchSize'] = 128
        args['learningRate'] = 0.001
        args['dropout'] = 0.9
        args['clip'] = 5.0

        args['encunit'] = 'lstm'
        args['decunit'] = 'lstm'
        args['enc_numlayer'] = 2
        args['dec_numlayer'] = 2

        args['maxLengthEnco'] = args['maxLength']
        args['maxLengthDeco'] = args['maxLength'] + 1

        args['temperature'] =1.0

        args['paratype'] = 'recog'


        return args

args = HP().args