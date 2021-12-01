def get_sentences_instance_concept(tokenizer, prop, concept_data):
    
    if prop != 'female':
        d_prop_type = [d for d in list(concept_data.values()) 
                     if 'property_type'in d]
        prop_type = d_prop_type[0]['property_type']
    else:
        prop_type = 'gender'
    
    mask = tokenizer.mask_token
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    
    if prop_type == 'complex' and prop == 'used_in_cooking':
        sentence = f"{cls_token} I used the {mask} to cook something. {sep_token}"
        control = f"{cls_token} I used the {mask} to {mask} something. {sep_token}"
        pl = False
        cap = False
    elif prop_type == 'gender':
        sentence = f"{cls_token} The {mask} showed herself. {sep_token}"
        control = f"{cls_token} The {mask} showed {mask}. {sep_token}"
        pl = False
        cap = False
    elif prop_type not in ['parts', 'activities']:
        sentence = f"{cls_token} The {mask} is {prop}. {sep_token}"
        control = f"{cls_token} The {mask} is {mask}. {sep_token}"
        pl = False
        cap = False
    elif prop_type == 'activities' and prop == 'lay_eggs':
        sentence = f"{cls_token} The {mask} lays eggs. {sep_token}"
        control = f"{cls_token} The {mask} lays {mask}. {sep_token}"
        pl = True
        cap = False
    elif prop_type == 'activities' and prop == 'fly':
        sentence = f"{cls_token} The  {mask} flew. {sep_token}"
        control = f"{cls_token} The  {mask} {mask}. {sep_token}"
        pl = False
        cap = False
    elif prop_type == 'activities' and prop == 'swim':
        sentence = f"{cls_token} The  {mask} swam. {sep_token}"
        control = f"{cls_token} The  {mask} {mask}. {sep_token}"
        pl = False
        cap = False
    elif prop_type == 'activities' and prop == 'roll':
        sentence = f"{cls_token} The  {mask} rolled. {sep_token}"
        control = f"{cls_token} The  {mask} {mask}. {sep_token}"
        pl = False
        cap = False
    elif prop_type == 'parts' and prop != 'made_of_wood':
        sentence = f" {cls_token} The {mask} has {prop}. {sep_token}"
        control = f"{cls_token} The {mask} has {mask}. {sep_token}"
        pl = False
        cap = False
    elif prop_type == 'parts' and prop == 'made_of_wood':
        sentence = f"{cls_token} The {mask} is made of wood. {sep_token}"
        control = f"{cls_token} The  {mask} is made of {mask}. {sep_token}"
        pl = False
        cap = False
    else:
        sentence = 'not found'
        control = 'not found'
        pl = False
        cap = False
    prop_s = '-'
    return sentence, control, prop_s


def get_sentences_instance_prop(tokenizer, prop, concept_data):
    
    if prop != 'female':
        d_prop_type = [d for d in list(concept_data.values()) 
                     if 'property_type'in d]
        prop_type = d_prop_type[0]['property_type']
    else:
        prop_type = 'gender'
    
    mask = tokenizer.mask_token
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    
    if prop_type == 'complex' and prop == 'used_in_cooking':
        sentence = f"{cls_token} I used the [concept] to {mask} something. {sep_token}"
        control = f"{cls_token} I used the {mask} to {mask} something. {sep_token}"
        pl = False
        prop_s = 'cook'
        cap = False
    elif prop_type == 'gender':
        sentence = f"{cls_token} The [concept] showed {mask}. {sep_token}"
        control = f"{cls_token} The {mask} showed {mask}. {sep_token}"
        pl = False
        prop_s = 'herself'
        cap = False
    elif prop_type not in ['parts', 'activities']:
        sentence = f"{cls_token} The [concept] is {mask}. {sep_token}"
        control = f"{cls_token} The {mask} is {mask}. {sep_token}"
        pl = False
        prop_s = prop
        cap = False
    elif prop_type == 'activities' and prop == 'lay_eggs':
        sentence = f"{cls_token} The [concept] lays {mask}. {sep_token}"
        control = f"{cls_token} The {mask} lays {mask}. {sep_token}"
        pl = False
        prop_s = 'eggs'
        cap = False
    elif prop_type == 'activities' and prop == 'fly':
        sentence = f"{cls_token} The [concept] {mask}. {sep_token}"
        control = f"{cls_token} The {mask} {mask}. {sep_token}"
        pl = False
        prop_s = 'flew'
        cap = False
    elif prop_type == 'activities' and prop == 'swim':
        sentence = f"{cls_token} The [concept] {mask}. {sep_token}"
        control = f"{cls_token} The {mask} {mask}. {sep_token}"
        pl = False
        prop_s = 'swam'
        cap = False
    elif prop_type == 'activities' and prop == 'roll':
        sentence = f"{cls_token} The [concept] {mask}. {sep_token}"
        control = f"{cls_token} The {mask} {mask}. {sep_token}"
        pl = False
        prop_s = 'rolled'
        cap = False
    elif prop_type == 'parts' and prop != 'made_of_wood':
        sentence = f"{cls_token} The [concept] has {mask}. {sep_token}"
        control = f"{cls_token} The {mask} has {mask}. {sep_token}"
        pl = False
        prop_s = prop
        cap = False
    elif prop_type == 'parts' and prop == 'made_of_wood':
        sentence = f"{cls_token} The [concept] is made of {mask}. {sep_token}"
        control = f"{cls_token} The {mask} is made of {mask}. {sep_token}"
        pl = False
        prop_s = 'wood'
        cap = False
    else:
        sentence = 'not found'
        control = 'not found'
        pl = False
        cap = False
    #return sentence, control, pl, prop_s, cap
    return sentence, control, prop_s