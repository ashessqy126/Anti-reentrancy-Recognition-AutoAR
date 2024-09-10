import json


with open('test.json', 'w') as f:
    json.dump(list({(1,2), (2,3,4)}), f)