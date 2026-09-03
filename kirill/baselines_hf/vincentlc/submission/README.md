# Random Baseline Submission

This is an example submission for the CVPR DeepFake Detection Challenge.

The model outputs a random score between 0 and 1 for every input image.

## Interface

Required interface:
Model(device, model_weights)
predict(images) -> torch.Tensor


Input:
- `images`: torch.Tensor [B, C, H, W] in [0,1] 

Output:
- `scores`: torch.Tensor [B] scores in [0,1]

## Files
submission.zip
├─ README.md
├─ paper/
│  └─ method.pdf  [TODO]
├─ src/
│  ├─ base_model.py    
│  ├─ model.py    [TODO]              
│  └─ anything else [TODO]
├─ weights/
│     └─ model.pt (или .pth/.safetensors) [TODO]
├─ docker/
|  └─ Dockerfile
│  └─ requirements.txt [TODO]
eval
