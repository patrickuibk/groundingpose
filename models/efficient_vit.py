from mmdet.registry import MODELS
from mmengine.model import BaseModule
from mmengine.logging import MMLogger, print_log
from mmengine.runner.checkpoint import CheckpointLoader

from efficientvit.models.efficientvit.backbone import(
    efficientvit_backbone_b0, efficientvit_backbone_b1,
    efficientvit_backbone_b2, efficientvit_backbone_b3,
    efficientvit_backbone_l0, efficientvit_backbone_l1,
    efficientvit_backbone_l2, efficientvit_backbone_l3)


@MODELS.register_module()
class EfficientViTBackbone(BaseModule):
    """EfficientViT backbone.

    Args:
        name (str): Name of the backbone.
        out_indices (tuple): Indices of stages to return.
        kwargs (dict): Arguments for the backbone.
    """

    def __init__(self, name, out_indices=(2, 3, 4), init_cfg=None, **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.out_indices = out_indices

        if name == 'efficientvit_backbone_b0':
            self.backbone = efficientvit_backbone_b0(**kwargs)
        elif name == 'efficientvit_backbone_b1':
            self.backbone = efficientvit_backbone_b1(**kwargs)
        elif name == 'efficientvit_backbone_b2':
            self.backbone = efficientvit_backbone_b2(**kwargs)
        elif name == 'efficientvit_backbone_b3':
            self.backbone = efficientvit_backbone_b3(**kwargs)
        elif name == 'efficientvit_backbone_l0':
            self.backbone = efficientvit_backbone_l0(**kwargs)
        elif name == 'efficientvit_backbone_l1':
            self.backbone = efficientvit_backbone_l1(**kwargs)
        elif name == 'efficientvit_backbone_l2':
            self.backbone = efficientvit_backbone_l2(**kwargs)
        elif name == 'efficientvit_backbone_l3':
            self.backbone = efficientvit_backbone_l3(**kwargs)
        else:
            raise ValueError(f'Unknown backbone name: {name}')
        
    def forward(self, x):
        x = self.backbone(x)
        return [x[f'stage{i}'] for i in self.out_indices]
    
    def train(self, mode=True):
        super().train(mode)
        self.backbone.train(mode)

    def init_weights(self):
        """Initialize weights from init_cfg, supporting prefix stripping."""
        if self.init_cfg is None:
            return

        if self.init_cfg.get('type') != 'Pretrained':
            raise NotImplementedError('Only Pretrained init_cfg is supported.')

        checkpoint = self.init_cfg.get('checkpoint')
        if checkpoint is None:
            raise ValueError('init_cfg must specify a checkpoint path.')

        logger = MMLogger.get_current_instance()
        ckpt = CheckpointLoader.load_checkpoint(checkpoint, logger=logger, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # Strip 'image_encoder.backbone.' prefix
        new_state_dict = {}
        prefix = 'image_encoder.backbone.'
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_k = k[len(prefix):]
                new_state_dict[new_k] = v
            else:
                continue

        missing, unexpected = self.backbone.load_state_dict(new_state_dict, strict=False)
        if missing:
            print_log(f'Missing keys when loading EfficientViTBackbone: {missing}', logger='current')
        if unexpected:
            print_log(f'Unexpected keys when loading EfficientViTBackbone: {unexpected}', logger='current')

