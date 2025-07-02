import torch
import os
from model import Net

angRes = 5
upfactor = 2
height = 128
width = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
path = "log"
filename = os.path.join(path, f"DistgSSR_{upfactor}xSR_{angRes}x{angRes}.pth.tar") 

model = Net(angRes,upfactor).to(device)
model.eval()

check_point = torch.load(filename, map_location=device)
model.load_state_dict(check_point["state_dict"])
print("Model loaded.")

example_input = torch.rand(1, 1, height*angRes, width*angRes).to(device)
print("Example input created.")

traced_script_module = torch.jit.trace(model, example_input)
print("Model traced.")

torch.jit.save(traced_script_module, filename.replace('.pth.tar', '.pt'))
print("Model saved.")