import re
import torch
import subprocess
# check deepspeed installation
report = subprocess.check_output(['python3', '-m', 'deepspeed.env_report']).decode()
r = re.compile('.*ninja.*OKAY.*')
assert any(r.match(line) for line in report.splitlines()) == True, "DeepSpeed Inference not correct installed"

# check cuda and torch version
torch_version,_ = torch.__version__.split("+")
torch_version = ".".join(torch_version.split(".")[:2])
cuda_version = torch.version.cuda
r = re.compile(f'.*torch.*{torch_version}.*')
assert any(r.match(line) for line in report.splitlines()) == True, "Wrong Torch version"
r = re.compile(f'.*cuda.*{cuda_version}.*')
assert any(r.match(line) for line in report.splitlines()) == True, "Wrong Cuda version"
