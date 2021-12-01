import json
import os
import csv


def get_properties():
    properties = []
    for path in os.listdir('../data/aggregated_semantic_info_scalar/'):
        prop = path.split('.')[0]
        if 'female-' not in prop and prop != '':
            properties.append(prop)
    return properties

def load_prop_data(prop):
    
    path = f'../data/aggregated_semantic_info_scalar/{prop}.json'
    with open(path) as infile:
        concept_dict = json.load(infile)
    return concept_dict



def get_labels(concept_data): #, l_type='bin'):
    concept_label_dict = dict()
    for c, d in concept_data.items():
        l_ml = d['ml_label']
        #if not '_' in c:
        #if l_type == 'bin':
        if l_ml in ['all', 'some', 'all-some', 'few-some']:
            concept_label_dict[c] = 'pos'
        elif l_ml in ['few']:
            concept_label_dict[c] = 'neg'
#             elif l_type == 'subset':
#                 if l_ml in ['all','all-some']:
#                     concept_label_dict[c] = 'all'
#                 elif l_ml in  ['some',  'few-some']:
#                     concept_label_dict[c] = 'some'
#                 elif l_ml in ['few']:
#                     concept_label_dict[c] = 'few'
#                 elif l_ml in ['creative']:
#                     concept_label_dict[c] = 'creative'
    return concept_label_dict



def probs_to_file(model_name, prop, results, sentence, control, target):
    # make dir:
    res_dir = f'../results/{target}/{model_name}/probabilities'
    os.makedirs(res_dir, exist_ok=True)
    
    f_path = f'{res_dir}/{prop}.csv'
    header = results[0].keys()
    with open(f_path, 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames = header)
        writer.writeheader()
        for d in results:
            writer.writerow(d)
            

def top_5_to_file(model_name, target, results):
    res_dir = f'../results/{target}/{model_name}'
    res_path = f'{res_dir}/probabilities-top5.csv'
    os.makedirs(res_dir, exist_ok=True)
    header = results[0].keys()
    with open(res_path, 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames = header)
        writer.writeheader()
        for d in results:
            writer.writerow(d)
            
            

def load_prop_type():
    prop_type_dict = dict()
    path = '../data/property_types.csv'
    with open(path) as infile:
        data = list(csv.DictReader(infile))
    for d in data:
        prop = d['property']
        t = d['type']
        prop_type_dict[prop] = t
    return prop_type_dict