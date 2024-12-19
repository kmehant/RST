import guidance
import json
import transformers
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from tqdm import tqdm
import time


def cnt_spaces_left(text):
  return (len(text)-len(text.lstrip(' ')))

model = sys.argv[1]
model_name = model
# dataset = sys.argv[2]
# output_file = sys.argv[3]
# colbert_top = sys.argv[4]

tokenizer = AutoTokenizer.from_pretrained(model)
model_base = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")

with open('./data_with_ft.jsonl', 'r') as json_file:
  json_list = list(json_file)

contents = {}
for json_str in json_list:
  result = json.loads(json_str)
  if result['ansible_module'] in result:
    contents[result['ansible_module']] = result[result['ansible_module']]


class nl2structure:
    def __init__(self, model_class: str, prompt, task="prompting", schema=None, reference_module=None, template=None, token_healing=True):
        global tokenizer
        global model_base
        self.schema = schema
        self.reference_module = reference_module
        self.tokenizer = tokenizer

        self.task = task
        self.template = template
        self.prompt = prompt
        self.token_healing = token_healing
        self.model = guidance.llms.Transformers(model=model_base, tokenizer=self.tokenizer)
        global contents
#         guidance.llm = guidance.llms.transformers.MPTChat(model=model,tokenizer=self.tokenizer)


    def return_required_keys(self, contents):
            
        dic = contents[self.reference_module]
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
                
        return(key_schema) 
        
    def return_template_elements(self):
        
        global contents
        required_keys = self.return_required_keys(contents)

        all_keys = list(contents[self.reference_module]["doc"]["options"].keys())
        optional_keys = []
        for i in all_keys:
          if i not in required_keys:
            optional_keys.append(i)
        return required_keys, optional_keys
    
    def build_template(self, required_keys, optional_keys):
      if self.task == "ansible_yaml":

        # template for ansible is built as all the level 1 keys in select and select in geneach with max_iteration = no. of keys there is gen after every select for value generation 
        # required_keys, optional_keys = self.return_template_elements()
        max_iter = len(required_keys.keys()) + len(optional_keys)
        guidance_template = """{{#geneach "items" min_iterations=0 max_iterations=""" + str(max_iter) +  """ unique=unique}}{{#select "key" id=id}}"""
        for key in list(required_keys.keys()) + optional_keys[:-1]:
          guidance_template += " " + key + "{{or}}"

        guidance_template += " " + optional_keys[-1] + "{{/select}}:" + """{{gen "value" dic=options_available}}{{/geneach}}"""
        self.template = guidance_template

    def return_nested_for_optional(self,contents, optional_keys, module):
      optional_required_nested_keys = {}
      optional_optional_nested_keys = {}
      for key in optional_keys:
        optional_required_nested_keys[key] = []
        optional_optional_nested_keys[key] = []
        if "suboptions" in contents[module]["doc"]["options"][key]:
            for key_nested in contents[module]["doc"]["options"][key]["suboptions"]:
                if "required" in contents[module]["doc"]["options"][key]["suboptions"][key_nested] and contents[module]["doc"]["options"][key]["suboptions"][key_nested]["required"]:
                    optional_required_nested_keys[key].append(key_nested)
                else:
                    optional_optional_nested_keys[key].append(key_nested)

      return(optional_required_nested_keys, optional_optional_nested_keys)

    
    def __call__(self): 
      
      self.prompt = self.prompt.replace("{{", "\\{{")
      if self.template:
        final_prompt = self.prompt
      else:
        
        if self.schema and self.reference_module:

          
          if self.task == "ansible_yaml":
            module_line = self.prompt.split("\n")[-1]
            module_spaces = " "*(cnt_spaces_left(self.prompt) + 2)
            # if self.prompt.startswith("name"):
            #   module_spaces = ""
            # else:
            #   module_spaces = " "*(cnt_spaces_left(self.prompt) + 2)
            module_selection_prompt  = self.prompt + "\n" + module_spaces + """{{select "module" options=module_list cmd_name=True}}:"""
            self.reference_module = eval(self.reference_module)
            guidance.llms.Transformers.cache.clear()

            prog_module = guidance(module_selection_prompt, token_healing=self.token_healing, llm=self.model)
            module_opt = prog_module(module_list=self.reference_module)
            # print(module_opt)
            # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

            self.reference_module = guidance.library._select.selected_module
            guidance.library._select.selected_module = None
            guidance.llms.Transformers.cache.clear()
            required_keys, optional_keys = self.return_template_elements()

            self.build_template(required_keys, optional_keys)
            final_prompt = self.prompt + "\n" + module_spaces + self.reference_module + ":\n" + module_spaces + " " + self.template
            regex_pattern = ".*\n\s{" + str(cnt_spaces_left(self.prompt) + 3) + "}$"    #regex pattern and it's corresponding template to change.
            # regex_pattern = ".*\n    $"                                                                       #for ansible we need regex pattern for level 1 keys. Which is 2 more spaces than module name line
            options_available = {regex_pattern: self.template}
      # options_available = None
            
      optional_required_nested_keys, optional_optional_nested_keys = self.return_nested_for_optional(contents, optional_keys + list(required_keys.keys()), self.reference_module)

      task_details = {"schema_path": self.schema, "task": self.task, "module_name": self.reference_module, "options_available": options_available, "required_keys": required_keys, "optional_keys": optional_keys, "optional_optional_nested_keys": optional_optional_nested_keys, "optional_required_nested_keys": optional_required_nested_keys}
      # print(options_available)
      prog1 = guidance(final_prompt, token_healing=self.token_healing, module=self.reference_module, options=options_available, schema=self.schema, task_config=task_details, llm=self.model)
      out = prog1(options_available=options_available, unique=[1], id=1)
      return(str(out))

# ir_data_df = pd.read_parquet(dataset)
# ir_data_jsonl = ir_data_df.to_dict(orient="records")
cnt_invalid = 0
# output = []

guidance.llms.Transformers.cache.clear()
guidance.library._gen.cnt = 0
guidance.library._gen.end_indent_flag = 0
guidance.library._gen.missed_required_keys = []
guidance.library._select.interrupt = 0
guidance.library._select.selected_module = None
guidance.schema_path = None
guidance.module_name = None
guidance.task_details = None
guidance.options_available = None
guidance._program_executor.parse_tree = None
guidance.library._geneach.select_generated_id = {}
guidance.library._geneach.iterator = None
guidance.library._geneach.cur_iteration = None
pred_structure = ""
prompt = "- name: Change file ownership, group and permissions"
obj = nl2structure(schema = "./data_with_ft.jsonl", reference_module = "['ansible.builtin.file', 'ansible.builtin.command']", model_class="bigcode/starcoderbase-1b", prompt=prompt, task="ansible_yaml", template=False)
pred_structure = obj()
if "{{ge" in pred_structure:
    end_ind = pred_structure.find("{{ge")
    pred_structure = pred_structure[:end_ind]
pred_structure = pred_structure.rstrip()
print(str(pred_structure))
# output.append(ir_data_jsonl[i])
#   f.write(pred_structure)
#   f.write("-----------------------------------------") 
print("done")
# except Exception as e:
#   print("failed")
#   print(e)
#   # ir_data_jsonl[i]["output"] = str("")
#   # output.append(ir_data_jsonl[i])
#   cnt_invalid += 1
  # continue
# df_out = pd.DataFrame(output)
# df_out.to_parquet(output_file)
# print(f"file saved to {output_file}")
# print("# invalid genereations: ", cnt_invalid)