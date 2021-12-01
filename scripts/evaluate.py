import utils
import csv
import os
from collections import Counter, defaultdict
import random
import pandas as pd



def load_results(prop, model_name, target):
    path = f'../results/{target}/{model_name}/probabilities/{prop}.csv'
    #sentence_v, level = sentence_v.split('-')
    with open(path) as infile:
        data = list(csv.DictReader(infile, delimiter = ','))
    return data

def aggregate_results(data, run, mode):
    label_dict = defaultdict(list)
    if mode == 'random':
        random.seed(run)
        n_pos = len([d for d in data if d['label'] == 'pos'])
        # pick random pos and neg
        pos_random = random.sample(data, n_pos)
    for d in data:
        l = d['label']
        p_norm = float(d['prob_norm'])
        if mode == 'true_labels':
            label_assigned = l
        elif mode == 'random':
            if d in pos_random:
                #print('found d' )
                label_assigned = 'pos'
            else:
                label_assigned = 'neg'
        d['label_assigned'] = label_assigned
        label_dict[label_assigned].append(p_norm)
    d = dict()
    for l, probs in label_dict.items():
        #print(l, len(probs))
        # mean
        d[l] = sum(probs)/len(probs)
        # median 
        #d[f'{l}-median'] = median(probs)
    if 'pos' in d and 'neg' in d:
        d['diff-pos-neg'] = d['pos'] - d['neg']
    return d


def get_diff(prop, model_name, target):
    mode = 'true_labels'
    run =  0
    results = load_results(prop, model_name, target)
    diff_dict = aggregate_results(results, run, mode)
    return diff_dict

def get_diff_random(prop, model_name, target):
    
    diff_dict = dict()
    
    mode = 'random'
    runs = range(100)
    
    # calculate on the basis of results
    results = load_results(prop, model_name, target)
    diffs = []
    pos = []
    neg = []
    for run in runs:
        diff_dict = aggregate_results(results, run, mode)
        diffs.append(diff_dict['diff-pos-neg'])
        pos.append(diff_dict['pos'])
        neg.append(diff_dict['neg'])
    diff_dict['diff-pos-neg'] = max(diffs)
    diff_dict['pos']= max(pos)
    diff_dict['neg'] = max(neg)
    return diff_dict


def main():
    
    target = 'property'
    properties = utils.get_properties()
    model_names = ['bert-base-uncased']
    models_str = '-'.join(sorted(model_names))
    
    path_dir = f'../analysis/{target}'
    os.makedirs(path_dir, exist_ok=True)
    path = f'{path_dir}/{models_str}.csv'

    overview_dict = defaultdict(dict)
    for model_name in model_names:
        for prop in properties:
            diff_dict = get_diff(prop, model_name, target)
            diff_dict_random = get_diff_random(prop, model_name, target)
            diff_true_random = diff_dict['diff-pos-neg'] - diff_dict_random['diff-pos-neg']
            overview_dict[prop][model_name] = diff_true_random

    df = pd.DataFrame(overview_dict)
    df.T.sort_values('bert-base-uncased').to_csv(path)
    
    
if __name__ == '__main__':
    main()
    