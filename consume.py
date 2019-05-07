import requests, json
import pandas as pd
import numpy as np
from datetime import date as date
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice

ws = Workspace.from_config('config.json')
service = Webservice(ws, 'keras-lstm')
key, _ = service.get_keys()

test_sample = []
for i in range(1,30):
    test_sample.append(date(2019, 5, i).strftime('%Y-%m-%d'))

test_sample = json.dumps({"dates": test_sample})
test_sample = bytes(test_sample, encoding='utf8')
headers = {'Content-Type':'application/json', 'Authorization': 'Bearer ' + key}

resp = requests.post(service.scoring_uri, test_sample, headers=headers)
print(resp.text)