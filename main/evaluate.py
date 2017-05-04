import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from utils import lookupIDX

def get_seqs(p1, p2, words):
    p1 = p1.split()
    p2 = p2.split()
    X1 = []
    X2 = []
    for i in p1:
        X1.append(lookupIDX(words,i))
    for i in p2:
        X2.append(lookupIDX(words,i))
    return X1, X2

def get_correlation(model, words, f):
    f = open(f,'r')
    lines = f.readlines()
    preds = []
    golds = []
    seq1 = []
    seq2 = []
    ct = 0
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; p2 = i[1]; score = float(i[2])
        X1, X2 = get_seqs(p1, p2, words)
        seq1.append(X1)
        seq2.append(X2)
        ct += 1
        if ct % 100 == 0:
            x1,m1 = model.prepare_data(seq1)
            x2,m2 = model.prepare_data(seq2)
            scores = model.scoring_function(x1,x2,m1,m2)
            scores = np.squeeze(scores)
            preds.extend(scores.tolist())
            seq1 = []
            seq2 = []
        golds.append(score)
    if len(seq1) > 0:
        x1,m1 = model.prepare_data(seq1)
        x2,m2 = model.prepare_data(seq2)
        scores = model.scoring_function(x1,x2,m1,m2)
        scores = np.squeeze(scores)
        preds.extend(scores.tolist())
    return pearsonr(preds,golds)[0], spearmanr(preds,golds)[0]

def evaluate_all(model, words):
    prefix = "../data/"
    parr = []; sarr = []

    farr = ["annotated-ppdb-dev",
            "annotated-ppdb-test"]

    for i in farr:
        p,s = get_correlation(model, words, prefix + i)
        parr.append(p); sarr.append(s)

    s = ""
    for i,j,k in zip(parr,sarr,farr):
        s += str(i)+" "+str(j)+" "+k+" | "

    print s
