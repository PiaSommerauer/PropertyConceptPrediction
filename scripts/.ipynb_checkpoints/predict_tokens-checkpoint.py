import sentence_generation
import models
import utils
import token_probability
import csv
import os

def main():
    
    model_name = 'roberta-large'
    model, tokenizer = models.load_model(model_name)
    targets = ['concept'] #, 'property']
    #properties = utils.get_properties()
    properties = ['black']
    for target in targets:
        for prop in properties:
            concept_data = utils.load_prop_data(prop)
            concept_label_dict = utils.get_labels(concept_data)


            if target == 'concept':
                sent, cont, prop_s = sentence_generation.get_sentences_instance_concept(tokenizer, prop, concept_data)
                results = token_probability.predict_concept(model, tokenizer, sent, cont, prop_s, concept_label_dict)
            elif target == 'property':
                sent, cont, prop_s = sentence_generation.get_sentences_instance_prop(tokenizer, prop, concept_data)
                results = token_probability.predict_property(model, tokenizer,sent, cont, prop_s, concept_label_dict)
            utils.probs_to_file(model_name, prop, results, sent, cont, target)
        
if __name__ == '__main__':
    main()