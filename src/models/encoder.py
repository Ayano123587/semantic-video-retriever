"""Pinned CLIP inference with verified pretrained weights and loading diagnostics."""
import hashlib
import numpy as np
from src.config import MODEL_ID, MODEL_REVISION

WEIGHTS_SHA256='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'

class CLIPEncoder:
    def __init__(self,device='cpu'):
        import torch
        from transformers import CLIPModel,CLIPProcessor
        from huggingface_hub import hf_hub_download
        torch.set_num_threads(2)
        self.torch=torch;self.device=device
        try:
            weights=hf_hub_download(MODEL_ID,'pytorch_model.bin',revision=MODEL_REVISION,local_files_only=True)
        except OSError:
            weights=hf_hub_download(MODEL_ID,'pytorch_model.bin',revision=MODEL_REVISION)
        with open(weights,'rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
        if digest!=WEIGHTS_SHA256:raise ValueError('Pinned CLIP weight hash mismatch')
        try:
            model,info=CLIPModel.from_pretrained(MODEL_ID,revision=MODEL_REVISION,use_safetensors=False,
                output_loading_info=True,local_files_only=True)
        except OSError:
            model,info=CLIPModel.from_pretrained(MODEL_ID,revision=MODEL_REVISION,use_safetensors=False,
                output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Incomplete or incompatible pretrained CLIP loading')
        self.model=model.to(device).eval()
        try:
            self.processor=CLIPProcessor.from_pretrained(MODEL_ID,revision=MODEL_REVISION,use_fast=False,local_files_only=True)
        except OSError:
            self.processor=CLIPProcessor.from_pretrained(MODEL_ID,revision=MODEL_REVISION,use_fast=False)
        self.receipt={'model_id':MODEL_ID,'revision':MODEL_REVISION,'weights_sha256':digest,'loading_info':info}
    def _normalize(self,values):
        values=values/values.norm(dim=-1,keepdim=True)
        result=values.cpu().numpy().astype(np.float32)
        if not np.isfinite(result).all():raise ValueError('Nonfinite encoder features')
        return result
    def images(self,images):
        if not images:raise ValueError('No frames')
        inputs=self.processor(images=images,return_tensors='pt').to(self.device)
        with self.torch.inference_mode():return self._normalize(self.model.get_image_features(**inputs))
    def texts(self,queries):
        if not queries or any(not isinstance(q,str) or not q.strip() for q in queries):
            raise ValueError('Nonempty text queries required')
        inputs=self.processor(text=queries,padding=True,truncation=True,return_tensors='pt').to(self.device)
        with self.torch.inference_mode():return self._normalize(self.model.get_text_features(**inputs))
    def text(self,query):return self.texts([query])[0]
