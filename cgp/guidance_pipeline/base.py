import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def cnt_spaces_left(text):
  return (len(text)-len(text.lstrip(' ')))


model = sys.argv[1]
tokenizer_base = AutoTokenizer.from_pretrained(model)
model_base = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")

prompt = "- name: Change file ownership, group\n"


try:
  final_prompt = prompt
  input_ids = tokenizer_base(final_prompt, return_tensors="pt").input_ids.to("cuda")
  gen_tokens = model_base.generate(input_ids, num_beams=5, num_return_sequences=1, max_new_tokens=150)
  gen_tokens = gen_tokens.reshape(1, -1, gen_tokens.shape[-1])[0][0]
  gen_code = tokenizer_base.decode(gen_tokens, skip_special_tokens=True)
  print(gen_code)
except Exception as e:
  print(e)
