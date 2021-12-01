import sentence_generation
import models
import utils
import token_probability


def main():
    
    model_name = 'bert-base-uncased'
    model, tokenizer = models.load_model(model_name)
    target = 'concept'
    properties = utils.get_properties()
    
    results = []
    for prop in properties:
        concept_data = utils.load_prop_data(prop)
        concept_label_dict = utils.get_labels(concept_data)
        if target == 'property':
            sent, cont, prop_s = sentence_generation.get_sentences_instance_prop(tokenizer, prop, concept_data)
            res  = token_probability.get_top_property(model, tokenizer, prop, concept_label_dict, sent)
        elif target == 'concept':
            sent, cont, prop_s = sentence_generation.get_sentences_instance_concept(tokenizer, prop, concept_data)
            res = token_probability.get_top_concept(model, tokenizer, prop, concept_label_dict, sent)
        results.append(res)
    utils.top_5_to_file(model_name, target, results)
        
        
        
if __name__ == '__main__':
    main()