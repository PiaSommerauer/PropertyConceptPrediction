
import torch
from torch.nn import functional as F


def get_model_probs(sentence, control, model, tokenizer, target):
    if target == 'concept':
        position_control_mask = 0
    elif target == 'property':
        position_control_mask = 1
    sm = torch.nn.Softmax(dim=0)
    sm_control = torch.nn.Softmax(dim=0)
    mask = tokenizer.mask_token
    tokens = tokenizer.tokenize(sentence)
    tokens_control = tokenizer.tokenize(control)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    indexed_tokens_control = tokenizer.convert_tokens_to_ids(tokens_control)
    # there is only one mask - the same mask is relevant for the control sentence
    mask_id = [n for n, t in enumerate(tokens) if t.strip() == mask][0]
    mask_id_c = [n for n, t in enumerate(tokens_control) if t.strip() == mask][position_control_mask]
    #print(mask_id_c)
    # token indices to tensor
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor_control = torch.tensor([indexed_tokens_control])
    # get output layer of token ids
    outputs = model(tokens_tensor)[0]
    outputs_control = model(tokens_tensor_control)[0]
    # get predictions for mask vec
    predictions = outputs[0][mask_id]
    predictions_control = outputs_control[0][mask_id_c]
    # convert to probabilities using softmax
    probs =  sm(predictions) 
    probs_control =  sm_control(predictions_control) 
    return probs, probs_control



def get_probs_property(probs, prop, tokenizer):
    word_id = tokenizer.convert_tokens_to_ids(prop)
    prob = probs[word_id]
    return prob

def get_probs_concepts(probs, concepts, tokenizer):
    ex_prob_dict = dict()
#     if pl == True:
#         p = inflect.engine()
    for word in concepts:
#         if pl == True:
#             word_model = p.plural(word)
#         else:
#             word_model = word
        word_id = tokenizer.convert_tokens_to_ids(word)
        prob = probs[word_id]
        ex_prob_dict[word] = prob
    return ex_prob_dict 


def predict_property(model, tokenizer, sent, cont, prop_s, concept_label_dict):
    results = []
    target = 'property'
    for concept, label in concept_label_dict.items():
        sent_c = sent.replace('[concept]', concept)
        cont_c = cont.replace('[concept]', concept)
        probs, probs_control = get_model_probs(sent_c,cont_c, model, 
                                                     tokenizer, 
                                                     target)
        prob_concept = float(get_probs_property(probs, prop_s, tokenizer))
        prob_control = float(get_probs_property(probs_control, prop_s, tokenizer))
        res_d = dict()
        res_d['concept'] = concept
        res_d['prop_s'] = prop_s
        res_d['label'] = label
        res_d['prob'] = prob_concept
        res_d['prob_control'] = prob_control
        res_d['prob_norm'] = prob_concept - prob_control
        results.append(res_d)
    return results


def predict_concept(model, tokenizer, sent, cont, prop_s, concept_label_dict):
    results = []
    target = 'concept'
    probs, probs_control = get_model_probs(sent, cont, model, 
                                                     tokenizer, 
                                                     target)
    concept_probs = get_probs_concepts(probs, concept_label_dict.keys(), 
                                                         tokenizer)   
    concept_probs_control = get_probs_concepts(probs_control, concept_label_dict.keys(), 
                                                                 tokenizer)
    for concept, label in concept_label_dict.items():
        prob = float(concept_probs[concept])
        prob_control = float(concept_probs_control[concept])
        res_d = dict()
        res_d['concept'] = concept
        res_d['prop_s'] = prop_s
        res_d['label'] = label
        res_d['prob'] = prob
        res_d['prob_control'] = prob_control
        res_d['prob_norm'] = prob - prob_control
        results.append(res_d)
    return results


def fill_mask(sentence, model, tokenizer):
    sm = torch.nn.Softmax(dim=0)
    mask = tokenizer.mask_token
    tokens = tokenizer.tokenize(sentence)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    # there is only one mask
    mask_id = [n for n, t in enumerate(tokens) if t.strip() == mask][0]
    # token indices to tensor
    tokens_tensor = torch.tensor([indexed_tokens])
    outputs = model(tokens_tensor)[0]
    #print(outputs.shape)
    predictions = outputs[0][mask_id]
    top_5_tok_ids = torch.topk(predictions, 5).indices.tolist() #[0].tolist()
    # convert to probabilities using softmax
    probs =  sm(predictions) 
    probs_top_5 = []
    toks_top_5 = []
    for tid in top_5_tok_ids:
        prob = probs[tid].item()
        probs_top_5.append(prob)
        word = tokenizer.decode(tid)
        toks_top_5.append(word.replace(' ', ''))
    return toks_top_5, probs_top_5



def get_top_property(model, tokenizer, prop, concept_label_dict, sent):
    tok_prob_cnts = Counter()
    
    concept_data = utils.load_prop_data(prop)
    examples_pos = [c for c,  l in concept_label_dict.items() if l == 'pos']
    
    for concept in examples_pos:
        sent_c = sent.replace('[concept]', concept)
        top5_toks_c, top5_probs_c = fill_mask(sent_c, model, tokenizer)
        for t, prob  in zip(top5_toks_c, top5_probs_c):
            tok_prob_cnts[t]+= prob
    top5_toks  = []
    top5_probs = []
    for t,  prob in tok_prob_cnts.most_common(5):
        top5_toks.append(t)
        top5_probs.append(prob/len(examples_pos))
    res = dict()
    res['property'] = prop
    res['top-5'] = ' '.join(top5_toks)
    res['mean-prob'] = sum(top5_probs)/len(top5_probs)
    return res


def get_top_concept(model, tokenizer, prop, concept_label_dict, sent):
    

    #sentence = get_sentence(tokenizer, prop, concept_data)
    
    top5_toks, top5_probs = fill_mask(sent, model, tokenizer)
    #print(top5_toks) #, top5_probs)
    res = dict()
    res['property'] = prop
    res['top-5'] = ' '.join(top5_toks)
    res['mean-prob'] = sum(top5_probs)/len(top5_probs)
    return res