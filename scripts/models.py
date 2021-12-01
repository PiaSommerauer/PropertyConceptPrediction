from transformers import BertTokenizer, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
# from transformers import AutoModelWithLMHead
# from transformers import AutoTokenizer, AutoModel


def load_model(model_name):
    if model_name in ['bert-base-uncased', 'bert-large-uncased']:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)
    elif model_name in ['roberta-base', 'roberta-large']:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForMaskedLM.from_pretrained(model_name)
    elif model_name in ['bert-ft']:
        model_path = '/Users/piasommerauer/Data/bert/model_bert-ft/'
        model = BertForMaskedLM.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
    elif model_name in ['coref-bert-base']:
        #model_path = '/Users/piasommerauer/Data/bert/coref-bert-base/'
        tokenizer = AutoTokenizer.from_pretrained('nielsr/coref-bert-base')
        model = AutoModelWithLMHead.from_pretrained('nielsr/coref-bert-base')
    elif model_name in ['roberta-ft']:
        model_path = '/Users/piasommerauer/Data/bert/model_roberta-ft/'
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForMaskedLM.from_pretrained(model_path)
    elif model_name in ['coref-roberta-base']:
        tokenizer = AutoTokenizer.from_pretrained('nielsr/coref-roberta-base')
        model = AutoModelWithLMHead.from_pretrained('nielsr/coref-roberta-base')
    eval_dict = model.eval()
    return model, tokenizer