import guidance
import json
import transformers
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from tqdm import tqdm
import time

f = open("/raid/nlp/sameer/guidance_updated/results.txt", "a")

def cnt_spaces_left(text):
  return (len(text)-len(text.lstrip(' ')))

tokenizer_base = AutoTokenizer.from_pretrained("/raid/nlp/sameer/guidance_updated/gpt_neo_ft_tldr_epoch_2.0/epoch-2.0")
model_base = AutoModelForCausalLM.from_pretrained("/raid/nlp/sameer/guidance_updated/gpt_neo_ft_tldr_epoch_2.0/epoch-2.0").to("cuda:6")
# model_base.load_state_dict(torch.load("/raid/nlp/sameer/guidance_updated/random_split_ft.bin", map_location=torch.device('cuda:4')))

model = guidance.llms.Transformers(model=model_base, tokenizer=tokenizer_base, device="cuda")

inference_df = pd.read_parquet("/raid/nlp/sameer/guidance_updated/gptneo-1b-ft-colbert-top-1-const.parquet")
ir_data_df = pd.read_parquet("/raid/nlp/sameer/guidance_updated/tldr/ir/tldr_split_dataset_test_ir_inf_clean.parquet")
cnt_invalid = 0

for i in tqdm(range(654, 655)):

  module_list = ir_data_df['top_1_colbert_ft'][i]
  prompt = inference_df['q'][i]
  module_line = prompt.split("\n")[-1]
  module_spaces = " "*(cnt_spaces_left(prompt) + 2)
  
  if prompt.startswith("name"):
    module_spaces = ""
  else:
    module_spaces = " "*(cnt_spaces_left(prompt) + 2)

  module_selection_prompt  = prompt + "\n" + module_spaces + """{{select "module" options=module_list cmd_name=True}}:"""
  reference_module = eval(module_list)
  guidance.llms.Transformers.cache.clear()

  try:
    prog_module = guidance(module_selection_prompt, token_healing=True, llm=model)
    module_opt = prog_module(module_list=reference_module)
    final_prompt = prompt + "\n" + module_spaces + guidance.library._select.selected_module + ":\n" + module_spaces + " "
    input_ids = tokenizer_base(final_prompt, return_tensors="pt").input_ids.to("cuda")
    gen_tokens = model_base.generate(input_ids, num_beams=5, num_return_sequences=1, max_new_tokens=150)
    gen_tokens = gen_tokens.reshape(1, -1, gen_tokens.shape[-1])[0][0]

    gen_code = tokenizer_base.decode(gen_tokens, skip_special_tokens=True)
    print(gen_code)
  except:
     cnt_invalid += 1
     continue

  # if "{{ge" in pred_structure:
  #     end_ind = pred_structure.find("{{ge")
  #     pred_structure = pred_structure[:end_ind]
  # pred_structure = pred_structure.rstrip()
  #   # f.write(pred_structure)
  #   # f.write("-----------------------------------------") 
  # cnt_invalid += 1

print("# invalid genereations: ", cnt_invalid)