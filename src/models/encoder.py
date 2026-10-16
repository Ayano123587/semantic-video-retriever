import numpy as np
from src.config import MODEL_ID,MODEL_REVISION

class CLIPEncoder:
    def __init__(self,device='cpu'):
        import torch
        from transformers import CLIPModel,CLIPProcessor
        torch.set_num_threads(2)
        self.torch=torch;self.device=device
        try:
            self.model=CLIPModel.from_pretrained(MODEL_ID,revision=MODEL_REVISION,use_safetensors=False,local_files_only=True)
            self.processor=CLIPProcessor.from_pretrained(MODEL_ID,revision=MODEL_REVISION,use_fast=False,local_files_only=True)
        except OSError:
            self.model=CLIPModel.from_pretrained(MODEL_ID,revision=MODEL_REVISION,use_safetensors=False)
            self.processor=CLIPProcessor.from_pretrained(MODEL_ID,revision=MODEL_REVISION,use_fast=False)
        self.model=self.model.to(device).eval()
    def images(self,images):
        if not images:raise ValueError('No frames')
        inputs=self.processor(images=images,return_tensors='pt').to(self.device)
        with self.torch.inference_mode():values=self.model.get_image_features(**inputs)
        values=values/values.norm(dim=-1,keepdim=True)
        return values.cpu().numpy().astype(np.float32)
    def text(self,query):
        if not query.strip():raise ValueError('Query must not be empty')
        inputs=self.processor(text=[query],padding=True,truncation=True,return_tensors='pt').to(self.device)
        with self.torch.inference_mode():values=self.model.get_text_features(**inputs)
        values=values/values.norm(dim=-1,keepdim=True)
        return values.cpu().numpy()[0].astype(np.float32)
