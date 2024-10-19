

def validate_format(obj: str) -> bool:
    
    if not isinstance(obj, str):
        raise ValueError('Only string inputs are allowed!')
    
    if len(obj) == 0:
        raise ValueError('Only non-empty inputs are allowed!')

def check_word_count(email: str) -> bool:
    
    words = email.split()
    is_plausible = len(words) > 1 and len(words) < 400
    
    return is_plausible

def check_label_existance(label: str, tag2idx: dict) -> bool:
    
    return label in tag2idx


