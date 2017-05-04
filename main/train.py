import utils
import lasagne
import random
import numpy as np
import sys
import argparse
import cPickle
from models import models
from example import example

def get_lines(f):
    f = open(f,'r')
    lines = f.readlines()
    d = []
    for i in lines:
        i = i.split('\t')
        d.append(i[-1].strip().lower())
    return d

def get_data(lines):
    examples = []
    for i in lines:
        e = (example(i[0]), example(i[1]))
        examples.append(e)
    return examples

def str2bool(v):
    if v is None:
        return False
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ValueError('A type that was supposed to be boolean is not boolean.')

def str2learner(v):
    if v is None:
        return None
    if v.lower() == "adagrad":
        return lasagne.updates.adagrad
    if v.lower() == "adam":
        return lasagne.updates.adam
    raise ValueError('A type that was supposed to be a learner is not a learner.')

random.seed(1)
np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("-LC", help="Regularization on composition parameters", type=float, default=0.)
parser.add_argument("-LW", help="Regularization on embedding parameters", type=float, default=0.)
parser.add_argument("-outfile", help="Name of output file")
parser.add_argument("-batchsize", help="Size of batch", type=int, default=100)
parser.add_argument("-dim", help="Dimension of model", type=int, default=300)
parser.add_argument("-wordfile", help="Word embedding file")
parser.add_argument("-save", help="Whether to pickle model", default="False")
parser.add_argument("-margin", help="Margin in objective function", type=float, default=0.4)
parser.add_argument("-samplingtype", help="Type of Sampling used: MAX, MIX, or RAND", default="MAX")
parser.add_argument("-evaluate", help="Whether to evaluate the model during training", default="True")
parser.add_argument("-epochs", help="Number of epochs in training", type=int, default=10)
parser.add_argument("-eta", help="Learning rate", type=float, default=0.001)
parser.add_argument("-learner", help="Either AdaGrad or Adam", default="adam")
parser.add_argument("-outgate", help="Whether to have an outgate for models using LSTM",default="True")
parser.add_argument("-model", help="Which model to use between (bi)lstm, (bi)lstmavg, (bi)gran, or wordaverage")
parser.add_argument("-mode", help="Train on SimpWiki (default) or equivalent amount of tokens from PPDB (set to ppdb)", default="simpwiki")
parser.add_argument("-scramble", type=float, help="Rate of scrambling", default=0.5)
parser.add_argument("-dropout", type=float, help="Dropout rate", default=0.)
parser.add_argument("-word_dropout", type=float, help="Word dropout rate", default=0.)
parser.add_argument("-gran_type", type=int,  help="Type of GRAN model", default=1)
parser.add_argument("-sumlayer", help="Whether to use sum layer for bi-directional recurrent networks", default="False")
parser.add_argument("-max", type=int, help="Maximum number of examples to use (<= 0 means use all data)", default=0)
parser.add_argument("-loadmodel", help="Name of pickle file containing model", default=None)

params = parser.parse_args()
params.save = str2bool(params.save)
params.evaluate = str2bool(params.evaluate)
params.outgate = str2bool(params.outgate)
params.learner = str2learner(params.learner)

data = []

f1 = '../data/simple.aligned'
f2 = '../data/normal.aligned'

d1 = get_lines(f1)
d2 = get_lines(f2)

for i in range(len(d1)):
    data.append((d1[i],d2[i]))
data = get_data(data)

lengths = []
for i in data:
    lengths.append(len(i[0].phrase.split()))
    lengths.append(len(i[1].phrase.split()))

if params.max > 0:
    random.shuffle(data)
    data = data[0:params.max]

if params.mode == "ppdb":
    d = utils.get_data("../data/ppdb-XL-ordered-data.txt")
    random.shuffle(d)
    ct = 0
    for i in data:
        ct += len(i[0].phrase.split()) + len(i[1].phrase.split())
    data = []
    print ct
    idx = 0
    ct2 = 0
    while ct > 0:
        dd = d[idx]
        data.append(dd)
        v = len(dd[0].phrase.split()) + len(dd[1].phrase.split())
        ct -= v
        ct2 += v
        idx += 1
    print ct2

if params.wordfile:
    (words, We) = utils.get_wordmap(params.wordfile)

model = models(We, params)

if params.loadmodel:
    base_params = cPickle.load(open(params.loadmodel, 'rb'))
    lasagne.layers.set_all_param_values(model.final_layer, base_params)

print " ".join(sys.argv)
print "Num examples:", len(data)

model.train(data, words, params)
