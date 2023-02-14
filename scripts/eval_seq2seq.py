import os, sys, glob, json
import numpy as np
import argparse
import torch

from torchmetrics.text.rouge import ROUGEScore
rougeScore = ROUGEScore()
from bert_score import score

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

def get_bleu(recover, reference):
    return sentence_bleu([reference.split()], recover.split(), smoothing_function=SmoothingFunction().method4,)

def selectBest(sentences):
    selfBleu = [[] for i in range(len(sentences))]
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            score = get_bleu(s1, s2)
            selfBleu[i].append(score)
    for i, s1 in enumerate(sentences):
        selfBleu[i][i] = 0
    idx = np.argmax(np.sum(selfBleu, -1))
    return sentences[idx]

def diversityOfSet(sentences):
    selfBleu = []
    # print(sentences)
    for i, sentence in enumerate(sentences):
        for j in range(i+1, len(sentences)):
            # print(sentence, sentences[j])
            score = get_bleu(sentence, sentences[j])
            selfBleu.append(score)
    if len(selfBleu)==0:
        selfBleu.append(0)
    div4 = distinct_n_gram_inter_sent(sentences, 4)
    return np.mean(selfBleu), div4


def distinct_n_gram(hypn,n):
    dist_list = []
    for hyp in hypn:
        hyp_ngrams = []
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
        total_ngrams = len(hyp_ngrams)
        unique_ngrams = len(list(set(hyp_ngrams)))
        if total_ngrams == 0:
            return 0
        dist_list.append(unique_ngrams/total_ngrams)
    return  np.mean(dist_list)


def distinct_n_gram_inter_sent(hypn, n):
    hyp_ngrams = []
    for hyp in hypn:
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
    total_ngrams = len(hyp_ngrams)
    unique_ngrams = len(list(set(hyp_ngrams)))
    if total_ngrams == 0:
        return 0
    dist_n = unique_ngrams/total_ngrams
    return  dist_n

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--folder', type=str, default='', help='path to the folder of decoded texts')
    parser.add_argument('--mbr', action='store_true', help='mbr decoding or not')
    parser.add_argument('--sos', type=str, default='[CLS]', help='start token of the sentence')
    parser.add_argument('--eos', type=str, default='[SEP]', help='end token of the sentence')
    parser.add_argument('--sep', type=str, default='[SEP]', help='sep token of the sentence')
    parser.add_argument('--pad', type=str, default='[PAD]', help='pad token of the sentence')

    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.folder}/*json"))
    sample_num = 0
    with open(files[0], 'r') as f:
        for row in f:
            sample_num += 1

    sentenceDict = {}
    referenceDict = {}
    sourceDict = {}
    for i in range(sample_num):
        sentenceDict[i] = []
        referenceDict[i] = []
        sourceDict[i] = []

    div4 = []
    selfBleu = []

    for path in files:
        print(path)
        sources = []
        references = []
        recovers = []
        bleu = []
        rougel = []
        avg_len = []
        dist1 = []

        with open(path, 'r') as f:
            cnt = 0
            for row in f:

                source = json.loads(row)['source'].strip()
                reference = json.loads(row)['reference'].strip()
                recover = json.loads(row)['recover'].strip()
                source = source.replace(args.eos, '').replace(args.sos, '')
                reference = reference.replace(args.eos, '').replace(args.sos, '').replace(args.sep, '')
                recover = recover.replace(args.eos, '').replace(args.sos, '').replace(args.sep, '').replace(args.pad, '')

                sources.append(source)
                references.append(reference)
                recovers.append(recover)
                avg_len.append(len(recover.split(' ')))
                bleu.append(get_bleu(recover, reference))
                rougel.append(rougeScore(recover, reference)['rougeL_fmeasure'].tolist())
                dist1.append(distinct_n_gram([recover], 1))

                sentenceDict[cnt].append(recover)
                referenceDict[cnt].append(reference)
                sourceDict[cnt].append(source)
                cnt += 1
                
        P, R, F1 = score(recovers, references, model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)

        print('*'*30)
        print('avg BLEU score', np.mean(bleu))
        print('avg ROUGE-L score', np.mean(rougel))
        print('avg berscore', torch.mean(F1))
        print('avg dist1 score', np.mean(dist1))
        print('avg len', np.mean(avg_len))

    if len(files)>1:
        if not args.mbr:
            print('*'*30)
            print('Compute diversity...')
            print('*'*30)
            for k, v in sentenceDict.items():
                if len(v) == 0:
                    continue
                sb, d4 = diversityOfSet(v)
                selfBleu.append(sb)
                div4.append(d4)

            print('avg selfBleu score', np.mean(selfBleu))
            print('avg div4 score', np.mean(div4))
        
        else:
            print('*'*30)
            print('MBR...')
            print('*'*30)
            bleu = []
            rougel = []
            avg_len = []
            dist1 = []
            recovers = []
            references = []
            sources = []


            for k, v in sentenceDict.items():
                if len(v) == 0 or len(referenceDict[k]) == 0:
                    continue

                recovers.append(selectBest(v))
                references.append(referenceDict[k][0])
                sources.append(sourceDict[k][0])

            for (source, reference, recover) in zip(sources, references, recovers):
                bleu.append(get_bleu(recover, reference))
                rougel.append(rougeScore(recover, reference)['rougeL_fmeasure'].tolist())
                avg_len.append(len(recover.split(' ')))
                dist1.append(distinct_n_gram([recover], 1))

            # print(len(recovers), len(references), len(recovers))
            
            P, R, F1 = score(recovers, references, model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)

            print('*'*30)
            print('avg BLEU score', np.mean(bleu))
            print('avg ROUGE-l score', np.mean(rougel))
            print('avg berscore', torch.mean(F1))
            print('avg dist1 score', np.mean(dist1))