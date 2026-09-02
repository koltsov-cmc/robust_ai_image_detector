from src.model import Model
m = Model(device='cuda', model_data_dir='weights')
print('OK, resolution', m.resolution, 'mean', m.mean, 'std', m.std)
