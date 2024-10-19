from typing import List
from pydantic import Field
from pydantic import validator
from pydantic import BaseModel


def validate_chars(email):
    
    return all(str(char).isalnum() or str(char).isspace() for char in email)


class SinglePredict(BaseModel):
    
    email: str = Field(description="Single email, represented as bag of words")

    @validator('email')
    def _validate_chars(cls, email):
        
        if not validate_chars(email):
            raise ValueError('Email must contain only alphanumeric characters and spaces!')
        
        return email

class BatchPredict(BaseModel):
    
    emails: List[str] = Field(min_items=1, description="List of emails, each represented as bag of words")
    
    @validator('emails')
    def _validate_chars(cls, emails):
        
        for n, email in enumerate(emails):
            if not validate_chars(email):
                raise ValueError('Non-alphanumeric characters or spaces detected in email {}!'.format(n))

        return emails
