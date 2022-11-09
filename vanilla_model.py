import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from measure_latency import measure_latency
from os import path
# Model Repository on huggingface.co
model_id = "../gpt-j-6B-fp16-sharded"

# Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = torch.load(path.join(model_id,'gptj-sharded.pt'),map_location="cuda:0")
print(f"model is loaded on device {model.device.type}")
# model is loaded on device cuda


payload="Hello my name is Philipp. I am getting in touch with you because i didn't get a response from you. What do I need to do to get my new card which I have requested 2 weeks ago? Please help me and answer this email in the next 7 days. Best regards and have a nice weekend but it"*2
print(f'Payload sequence length is: {len(tokenizer(payload)["input_ids"])}')

# generation arguments
generation_args = dict(
  do_sample=False,
  num_beams=1,
  min_length=128,
  max_new_tokens=128
)
vanilla_results = measure_latency(model=model,tokenizer=tokenizer,payload=payload,generation_args=generation_args, device=model.device)

print(f"Vanilla model: {vanilla_results[0]}")
