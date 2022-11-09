from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "../gpt-j-6B-fp16-sharded",
    torch_dtype=torch.float16,
)


torch.save(model, "gptj-sharded.pt")
