# import guidance
import json
import transformers
import sys
import yaml
import pandas as pd

def return_required_keys_(contents, module):  #function to return the required keys from schema
    dic = contents[module]
    required_keys = []
    key_schema = {}
    for key in dic["doc"]["options"]:
        if "required" in dic["doc"]["options"][key].keys():
            if dic["doc"]["options"][key]["required"]==True:
                required_keys.append(key)
                
    for key in required_keys:
        key_schema[key] = []
        if "suboptions" in dic["doc"]["options"][key]:
            for key_1 in dic["doc"]["options"][key]["suboptions"]:
                if "required" in dic["doc"]["options"][key]["suboptions"][key_1] and dic["doc"]["options"][key]["suboptions"][key_1]["required"]==True:
                    key_schema[key].append(key_1)
                
    return(key_schema)   #return type is a dictionalry with key as required key names

schema_path = "/raid/nlp/sameer/guidance-guidanceV2/data_with_ft.jsonl"

# with open(schema_path, 'r') as j:   #read main schema file
#      contents = json.loads(j.read())

with open('/raid/nlp/sameer/guidance-guidanceV2/data_with_ft.jsonl', 'r') as json_file:
    json_list = list(json_file)

contents = {}
for json_str in json_list:
    result = json.loads(json_str)
    if result['ansible_module'] in result:
        contents[result['ansible_module']] = result[result['ansible_module']]
    else:
        print(result)
print(return_required_keys_(contents, "azure.azcollection.azure_rm_virtualmachine"))
