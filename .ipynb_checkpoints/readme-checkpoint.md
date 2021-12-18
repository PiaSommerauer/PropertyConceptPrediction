# Contrastive masked token prediction for pre-trained language models

Do positive examples of a property yield higher token probabilities than negative examples in sentences that evoke the propertty/concept?

Positive example: The lemon is yellow. --> The [mask] is yellow.  / The lemon is [mask].

Negative example: The sea is yellow. --> The [mask] is yellow. / The sea is [mask].



1.) Run predict_top.py to validate the templates

2.) Run predict_tokens.py to get the concept and property probabilities

3.) Run evaluate.py to evaluate. 

Evaluation:

Difference of mean probabilities of positive examples and mean probabilities of negative examples

Comparison against random:

Is the observed difference higher than the maximal difference over 100 random label distributions?