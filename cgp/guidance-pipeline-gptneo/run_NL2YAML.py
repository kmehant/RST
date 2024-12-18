import guidance
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


def cnt_spaces_left(text):
  return (len(text)-len(text.lstrip(' ')))


# tokenizer = transformers.AutoTokenizer.from_pretrained("/dccstor/ai4code-ansible/shared/model_delivery/wisdom-ansible-v9/model_20230326-135203/checkpoint")
# guidance.llm = guidance.llms.transformers.MPTChat(model="/dccstor/ai4code-ansible/shared/model_delivery/wisdom-ansible-v9/model_20230326-135203/checkpoint",tokenizer=tokenizer)


def NL2YAML(prompt, module_name, schema_path, module_list, template=None):

  with open('/dccstor/rhassistant/mehant/paper/guidance/guidance-guidanceV2/data_with_ft.jsonl', 'r') as json_file:
      json_list = list(json_file)

  contents = {}
  for json_str in json_list:
      result = json.loads(json_str)
      if result['ansible_module'] in result:
        contents[result['ansible_module']] = result[result['ansible_module']]
      else:
        print(result)

  if(template):   #if template given by user
    prog1 = guidance(template, token_healing=True, module=module_name)
  else:
    guidance_template = ""
    module_line = prompt.split("\n")[-1]
    # guidance_template = """{{#geneach "items" min_iterations=0 max_iterations=10 unique=unique}}{{#select "key" id=id}}"""
    # guidance_template = prompt + "\n" + " "*cnt_spaces_left(module_line) + "  " + """azure.azcollection.azure_rm_virtualmachine:""" + "\n"

    module_name = guidance.library._select.selected_module
    required_keys = return_required_keys_(contents, module_name)
    all_keys = list(contents[module_name]["doc"]["options"].keys())

    optional_keys = []
    for i in all_keys:
        if i not in required_keys:
          optional_keys.append(i)

    guidance_template += """{{#geneach "items" min_iterations=0 max_iterations=10 unique=unique}}{{#select "key" id=id}}"""

    for key in list(required_keys.keys()) + optional_keys[:-1]:
        guidance_template += key + "{{or}}"

    guidance_template += optional_keys[-1] + "{{/select}}" + """{{gen "value" dic=options_available}}{{/geneach}}"""

    print(guidance_template)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    regex_pattern = ".*\n\s{" + str(cnt_spaces_left(module_line) + 2) + "}$"    #regex pattern and it's corresponding template to change
    prompt_ = prompt + "\n" + " "*cnt_spaces_left(module_line) + "  " + """azure.azcollection.azure_rm_virtualmachine:""" + "\n" + " "*cnt_spaces_left(module_line) + "    " + guidance_template 
    options_available = {regex_pattern: guidance_template}
    print(prompt_)
    prog1 = guidance(prompt_, token_healing=True, module=module_name, options=options_available, module_list=module_list)
  # out = prog1(options_available=options_available, unique=[1], id=1)
  # with open("yaml.txt", "w") as file1:
  #   # Writing data to a file
  #   file1.write(str(out))

  # print(out)


# data = []
# with open("/Users/sameerp30/Downloads/data_with_ft.jsonl","r") as f1:
#     lines = f1.readlines()
#     for line in lines:
#         data.append(json.loads(line))

# for i in range(0,len(data)):
#     prompt = data[i]["input_script"] + "\n".join(data[i]["output_script"][i].split("\n")[0:2])
#     module = data[i]['Preferred module']

schema_path = "/dccstor/rhassistant/mehant/paper/guidance/guidance-guidanceV2/data_with_ft.jsonl"

inference_df = pd.read_parquet("/raid/nlp/sameer/guidance-guidanceV2/tldr/tldr/base_models/bigcode_starcoderbase-1b.parquet")
ir_data_df = pd.read_parquet("/raid/nlp/sameer/guidance-guidanceV2/random_85_15_split_test_ir_inf.parquet")
prompt = inference_df['c'][0] + "\n" + inference_df['q'][0]
module_list = ir_data_df['top_3_colbert_ft'][0]
NL2YAML(prompt, "azure.azcollection.azure_rm_virtualmachine", schema_path, module_list)
print(ir_data_df['top_3_colbert_ft'][0])
print(ir_data_df.columns)


