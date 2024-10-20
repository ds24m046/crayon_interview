import os
from pymongo import MongoClient
from datetime import datetime as dt


def get_collection(collection_name='emails'):
    
    default = 'mongodb://localhost:27017/emails'
    mongo_path = os.getenv('MONGO_URI', default)

    client = MongoClient(mongo_path)
    database = client['emails']
    collection = database[collection_name]
    
    return collection

def get_emails(collection, limit=1000):
    
    emails = list(collection.find().limit(limit))
    emails = [i['email'] for i in emails]
    
    return emails

def get_drifts(collection, limit=1000):
    
    records = list(collection.find().limit(limit))
    
    drifts = []
    for record in records:
        drift = {key: record[key] for key in ('timestamp', 'type', 'metric')}
        drifts.append(drift)
        
    return drifts

def save_email(email, collection):
    
    record = {
        'timestamp': dt.now().isoformat(),
        'email': email
    }

    result = collection.insert_one(record)
    
    return result.inserted_id

def save_drift(drift, collection):
    
    record = {'timestamp': dt.now().isoformat()}
    
    for key, val in drift.items():
        record[key] = val
    
    result = collection.insert_one(record)
    
    return result.inserted_id
