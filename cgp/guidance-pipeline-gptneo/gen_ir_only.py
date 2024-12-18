import guidance
import json
import transformers
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from tqdm import tqdm
import time

# f = open("/raid/nlp/sameer/guidance_updated/results.txt", "a")

def cnt_spaces_left(text):
  return (len(text)-len(text.lstrip(' ')))


model = sys.argv[1]
dataset = sys.argv[2]
output_file = sys.argv[3]
colbert_top = sys.argv[4]


tokenizer_base = AutoTokenizer.from_pretrained(model)
model_base = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")
model = guidance.llms.Transformers(model=model_base, tokenizer=tokenizer_base, device="cuda")
ir_data_df = pd.read_parquet(dataset)
ir_data_jsonl = ir_data_df.to_dict(orient="records")
cnt_invalid = 0
output = []

for i in tqdm(range(0, len(ir_data_df))):

  module_list = ir_data_df[f'top_{colbert_top}_colbert_ft'][i]
  prompt = ir_data_df['q'][i]
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
    ir_data_jsonl[i]["output"] = str(gen_code)
    output.append(ir_data_jsonl[i])
  except:
    ir_data_jsonl[i]["output"] = ""
    output.append(ir_data_jsonl[i])
    cnt_invalid += 1
              
df_out = pd.DataFrame(output)
df_out.to_parquet(output_file)

print("# invalid genereations: ", cnt_invalid)