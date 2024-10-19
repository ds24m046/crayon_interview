from nltk.stem import PorterStemmer


def remove_bad_words(email: str) -> str:
    
    if not isinstance(email, str):
        raise ValueError('email should be string')
    
    words = email.split()
    words = [i for i in words if len(i)>2 and len(i)<20]
    
    return ' '.join(words)

def stem_words(email: str) -> str:

    words = email.split()
    stems = [PorterStemmer().stem(i) for i in words]
    
    return ' '.join(stems)