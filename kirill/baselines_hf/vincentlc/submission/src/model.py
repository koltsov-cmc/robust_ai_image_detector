import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoImageProcessor
from torchvision.transforms import Resize, Normalize

from .base_model import BaseDeepFakeModel

# class HFBackboneWrapper(nn.Module):
#     def __init__(self, model_name_or_path, pretrained=False, freeze_backbone=False):
#         super().__init__()
#         self.config = AutoConfig.from_pretrained(model_name_or_path)
#         self.config.output_hidden_states = True 
        
#         self.model = AutoModel.from_config(self.config)
        
#         if hasattr(self.model, 'vision_model'):
#             self.vision_model = self.model.vision_model
#         else:
#             self.vision_model = self.model

#         if hasattr(self.config, 'vision_config'):
#             self.feature_dim = getattr(self.config.vision_config, 'hidden_size')
#         else:
#             self.feature_dim = getattr(self.config, 'hidden_size')

#         if freeze_backbone:
#             for param in self.model.parameters():
#                 param.requires_grad = False

#     def forward(self, x):
#         outputs = self.vision_model(x, output_hidden_states=True)
#         return outputs.hidden_states

class HFBackboneWrapper(nn.Module):
    def __init__(self, model_name_or_path, pretrained=False, freeze_backbone=False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.config.output_hidden_states = True

        # ВАЖНО: если нужен pretrained backbone:
        # self.model = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.model = AutoModel.from_config(self.config)

        if hasattr(self.model, "vision_model"):
            self.vision_model = self.model.vision_model
        else:
            self.vision_model = self.model

        if hasattr(self.config, "vision_config"):
            self.feature_dim = self.config.vision_config.hidden_size
        else:
            self.feature_dim = self.config.hidden_size

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x):
        # явный pixel_values + return_dict
        outputs = self.vision_model(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True
        )

        hs = getattr(outputs, "hidden_states", None)
        if hs is None:
            hs = (outputs.last_hidden_state,)
        return hs

class MeanPoolingAggregator(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.out_dim = feature_dim

    def forward(self, hidden_states):
        last_layer = hidden_states[-1]
        return last_layer.mean(dim=1)

class AdvancedClassifier(nn.Module):
    def __init__(self, backbone, aggregator, num_classes):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.head = nn.Linear(aggregator.out_dim, num_classes)

    def forward(self, x):
        all_features = self.backbone(x)
        feat_vec = self.aggregator(all_features)
        logits = self.head(feat_vec)
        return logits

class Model(BaseDeepFakeModel):
    def __init__(self, device, model_data_dir):
        super().__init__(device)
        self.device = device
        
        self.local_siglip_dir = os.path.join(model_data_dir, "siglip_config")
        
        self.ckpt_name = "best_model.pth" 
        self.ckpt_path = os.path.join(model_data_dir, self.ckpt_name)
        
        self.resolution = 384
        self.num_classes = 2
        
        try:
            processor = AutoImageProcessor.from_pretrained(self.local_siglip_dir)
            self.mean = processor.image_mean
            self.std = processor.image_std
            self.resample = getattr(processor, 'resample', 3) 
        except Exception as e:
            print(f"Warning: Failed to load processor from {self.local_siglip_dir}. Using default SigLIP values. Error: {e}")
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
            from torchvision.transforms import InterpolationMode
            self.resample = InterpolationMode.BICUBIC

        self.resize = Resize((self.resolution, self.resolution), interpolation=self.resample, antialias=True)
        self.normalize = Normalize(mean=self.mean, std=self.std)

        backbone = HFBackboneWrapper(self.local_siglip_dir, pretrained=False, freeze_backbone=False)
        aggregator = MeanPoolingAggregator(feature_dim=backbone.feature_dim)
        self.net = AdvancedClassifier(backbone, aggregator, num_classes=self.num_classes)
        
        if os.path.exists(self.ckpt_path):
            state_dict = torch.load(self.ckpt_path, map_location="cpu")
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            
            self.net.load_state_dict(state_dict)
            print(f"Loaded weights from {self.ckpt_path}")
        else:
            print(f"WARNING: Checkpoint {self.ckpt_path} not found! Model will use random weights.")

        self.net.to(self.device)
        self.net.eval()

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: torch.Tensor [B, 3, H, W] in [0,1], on device

        Returns:
            torch.Tensor [B] scores in [0,1] (Probability of being AI-generated)
        """

        x = self.resize(images)
        x = self.normalize(x)
        
        with torch.no_grad():
            logits = self.net(x)
            
            probs = F.softmax(logits, dim=-1)
            generated_scores = probs[:, 1]
            
        return generated_scores