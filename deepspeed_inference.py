import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
from os import path
from measure_latency import measure_latency
from optimum.onnxruntime import ORTModelForCausalLM
from pathlib import Path
onnx_path = Path("onnx")
# Model Repository on huggingface.co
#model_id = "../model/gpt-j-6B-fp16-sharded/"
model_id = "philschmid/gpt-j-6B-fp16-sharded"
#model_id = "EleutherAI/gpt-j-6B"

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
#model = torch.load(path.join(model_id,'gptj-sharded.pt'),map_location='cuda:0')

# init deepspeed inference engine
ds_model = deepspeed.init_inference(
    model=model,      # Transformers models
    mp_size=1,        # Number of GPU
    dtype=torch.float16, # dtype of the weights (fp16)
    replace_method="auto", # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True, # replace the model with the kernel injector
)
print(f"model is loaded on device {ds_model.module.device}")
from deepspeed.ops.transformer.inference import DeepSpeedTransformerInference

assert isinstance(ds_model.module.transformer.h[0], DeepSpeedTransformerInference) == True, "Model not sucessfully initalized"

# Test model
example = "My name is Philipp and I"
input_ids = tokenizer(example,return_tensors="pt").input_ids.to(model.device)
logits = ds_model.generate(input_ids, do_sample=True, max_length=100)
print(tokenizer.decode(logits[0].tolist()))

payload = (
    "Hello my name is Philipp. I am getting in touch with you because i didn't get a response from you. What do I need to do to get my new card which I have requested 2 weeks ago? Please help me and answer this email in the next 7 days. Best regards and have a nice weekend but it"
    * 2
)
print(f'Payload sequence length is: {len(tokenizer(payload)["input_ids"])}')

# generation arguments
generation_args = dict(do_sample=False, num_beams=1, min_length=128, max_new_tokens=128)
ds_results = measure_latency(model=ds_model, tokenizer=tokenizer,payload=payload, generation_args=generation_args, device=ds_model.module.device)

print(f"DeepSpeed model: {ds_results[0]}")
