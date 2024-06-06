import re

import torch
from torch import nn

from src.model import Encoder

class Classifier(nn.Module):
    def __init__(self, num_classes=10, ckpt_path=None):
        super().__init__()
        self.clay_encoder = Encoder(
            mask_ratio=0.0,
            patch_size=8,
            shuffle=False,
            dim=768,
            depth=12,
            heads=12,
            dim_head=64,
            mlp_ratio=4.0,
        )
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.load_clay_weights(ckpt_path)

    def load_clay_weights(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get("state_dict")
    
        # Remove model.encoder prefix for the clay encoder
        state_dict = {
            re.sub(r"^model\.encoder\.", "", name): param
            for name, param in state_dict.items()
            if name.startswith("model.encoder")
        }
    
        for name, param in self.clay_encoder.named_parameters():
            if name in state_dict and param.size() == state_dict[name].size():
                param.data.copy_(state_dict[name])  # Copy the weights
            else:
                print(f"No matching parameter for {name} with size {param.size()}")
    
        # Freeze clay encoder
        for param in self.clay_encoder.parameters():
            param.requires_grad = False
    
        self.clay_encoder.eval()

    def forward(self, datacube):
        embeddings, *_ = self.clay_encoder(datacube) # embeddings: batch x (1 + row x col) x 768
        embeddings = embeddings[:, 0, :]
        logits = self.head(embeddings)
        return logits