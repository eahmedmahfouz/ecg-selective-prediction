import torch
ckpt = torch.load(
    "../ecg_ptbxl_benchmarking/output/exp0/models/fastai_xresnet1d101/models/fastai_xresnet1d101.pth",
    map_location="cpu",
    weights_only=False,
)
print(type(ckpt))
if isinstance(ckpt, dict):
    keys = list(ckpt.keys())
    print(len(keys), "keys")
    print(keys[:10])
    print(keys[-10:])
else:
    print(ckpt)
    
    
model_sd = ckpt["model"]
print(type(model_sd))
keys = list(model_sd.keys())
print(len(keys), "keys")
for k in keys[:15]:
    print(k, model_sd[k].shape)
print("...")
for k in keys[-15:]:
    print(k, model_sd[k].shape)
