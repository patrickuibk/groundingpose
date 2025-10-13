from typing import Dict, Sequence, Any
import torch
from mmdet.models.language_models.bert import BertModel
from mmdet.registry import MODELS



@MODELS.register_module()
class MemoizedBertModel(BertModel):
    """Wrapper class that memoizes a BertModel to avoid repeated computations.
    
    This wrapper caches the results of the language model based on input strings.
    When the same text inputs are provided, it returns the cached result instead
    of recomputing the embeddings.
    """
    
    def __init__(self, use_cache: bool=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        self.use_cache = use_cache

    def forward(self, captions: Sequence[str], **kwargs) -> Dict[str, Any]:
        """Memoized call to the BERT language model.
        
        Args:
            captions: List of strings to process
            
        Returns:
            Dict containing the language model outputs
        """
        with torch.no_grad():
            if not self.use_cache:
                return super().forward(captions, **kwargs)

            # TODO: Currently uses the whole batch as key, could be improved by using individual strings.

            key = tuple(captions)
            
            if key in self.cache:
                cached_result = {}
                for k, v in self.cache[key].items():
                    cached_result[k] = v.clone() if isinstance(v, torch.Tensor) else v
                return cached_result
            
            result = super().forward(captions, **kwargs)
            
            self.cache[key] = {
                k: v.detach().clone() if isinstance(v, torch.Tensor) else v 
                for k, v in result.items()
            }
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the cached results."""
        self.cache.clear()

