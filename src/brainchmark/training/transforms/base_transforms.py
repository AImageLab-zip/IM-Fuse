from abc import ABC, abstractmethod
from typing import Callable, Any
import torch

class TransformManager(ABC):
    def __init__(self):
        # Subclasses set self.train_transforms and self.test_transforms as callables
        self.train_transforms: dict[str, Callable] = {}
        self.test_transforms: dict[str, Callable] = {}
        self._setup_transforms()
    
    @abstractmethod
    def _setup_transforms(self):
        # Subclasses implement: hardcode the transforms here
        pass
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor, mode: str = 'train', *extra_tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if mode not in ['train', 'test']:
            raise ValueError("Mode must be 'train' or 'test'")
        
        transforms = self.train_transforms if mode == 'train' else self.test_transforms
        
        # Apply to images and labels (required)
        transformed_images = transforms['images'](images)
        transformed_labels = transforms['labels'](labels)
        # Apply to extras if defined (e.g., 'extra_0' key)
        transformed_extras = []
        for i, extra in enumerate(extra_tensors):
            key = f'extra_{i}'
            if key in transforms:
                transformed_extras.append(transforms[key](extra))
            else:
                transformed_extras.append(extra)  # No-op if not defined
        
        return (transformed_images, transformed_labels, *transformed_extras)