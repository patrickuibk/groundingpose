from typing import Dict, Tuple, List

import torch
from torch import Tensor
from torch import nn

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.models.detectors import GroundingDINO
from mmdet.models.detectors.glip import (create_positive_map, create_positive_map_label_to_token)

from .decoder import GroundingPOSETransformerDecoder


@MODELS.register_module()
class GroundingPOSE(GroundingDINO):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        decoder_cfg = self.decoder
        super()._init_layers()
        self.decoder = GroundingPOSETransformerDecoder(**decoder_cfg)
        self.text_feat_map_relations = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(tokenized, tokens_positive, self.bbox_head.max_text_len) # fixed max_text_len
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    
    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(img_feats, batch_data_samples)

        # combine original and relation text dict
        text_dict_relation = self.get_relation_text_dict(batch_data_samples)
        num_tokens_org, num_tokens_relation = text_dict['embedded'].shape[1], text_dict_relation['embedded'].shape[1]
        masks_combined = torch.zeros((text_dict['text_token_mask'].shape[0], num_tokens_org + num_tokens_relation, num_tokens_org + num_tokens_relation), device=text_dict['masks'].device, dtype=text_dict['masks'].dtype)
        masks_combined[:, :num_tokens_org, :num_tokens_org] = text_dict['masks']
        masks_combined[:, num_tokens_org:, num_tokens_org:] = text_dict_relation['masks']
        text_dict_combined = {
            'embedded': torch.cat([text_dict['embedded'], text_dict_relation['embedded']], dim=1),
            'hidden': torch.cat([text_dict['hidden'], text_dict_relation['hidden']], dim=1),
            'text_token_mask': torch.cat([text_dict['text_token_mask'], text_dict_relation['text_token_mask']], dim=1),
            'masks': masks_combined,
            'position_ids': torch.cat([text_dict['position_ids'], text_dict_relation['position_ids']], dim=1)
        }

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict, text_dict=text_dict_combined)

        # split original text memory and relation text memory
        memory_text_combined = encoder_outputs_dict['memory_text']
        text_token_mask_combined = encoder_outputs_dict['text_token_mask']
        memory_text_relation = memory_text_combined[:, num_tokens_org:, :]
        text_token_mask_relation = text_token_mask_combined[:, num_tokens_org:]
        encoder_outputs_dict['memory_text'] = memory_text_combined[:, :num_tokens_org, :]
        encoder_outputs_dict['text_token_mask'] = text_token_mask_combined[:, :num_tokens_org]

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict) 
        
        head_inputs_dict.update(dict(
            memory=encoder_outputs_dict['memory'],
            memory_mask=encoder_outputs_dict['memory_mask'],
            spatial_shapes=encoder_outputs_dict['spatial_shapes'],
            level_start_index=encoder_inputs_dict['level_start_index'],
            valid_ratios=encoder_inputs_dict['valid_ratios'],
            memory_relation_text=memory_text_relation,
            relation_text_token_mask=text_token_mask_relation
        ))
        
        return head_inputs_dict
    

    def get_relation_text_dict(self, batch_data_samples: SampleList) -> List[List[dict]]:
        """Get relation text dict for the relation encoder."""
        caption_strings = []
        for data_sample in batch_data_samples:

            if self.training:
                tokenized, caption_string, tokens_positive_per_relation_type, _ = self.get_tokens_and_prompts(data_sample.relation_text, True)

                # Shape: (num_relations, max_text_len)
                # [i, j] = 1 if relation i is associated with token j
                positive_map_per_relation_type = create_positive_map(tokenized, tokens_positive_per_relation_type, self.bbox_head.max_text_len)
                positive_map_per_relation_type = positive_map_per_relation_type.to(data_sample.gt_instances.relation_matrices.device)
                positive_map_per_relation_type = positive_map_per_relation_type.bool().float()

                data_sample.gt_positive_map_relation = positive_map_per_relation_type # (num_relations, max_text_len)
            else:
                token_positive_map_relation, caption_string, _, _ = self.get_tokens_positive_and_prompts(data_sample.relation_text, True)
                data_sample.token_positive_map_relation = token_positive_map_relation
                
            caption_strings.append(caption_string)

        
        relation_text_dict = self.language_model(caption_strings)
        if self.text_feat_map_relations is not None:
            relation_text_dict['embedded'] = self.text_feat_map_relations(relation_text_dict['embedded'])

        return relation_text_dict
    
    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)

        # Do not add denoising queries as in GroundingDINO
        dn_mask, dn_meta = None, {'num_denoising_queries': 0}
        reference_points = topk_coords.detach()

        # Discard width and height
        reference_points = reference_points[..., :2]
        if self.training:
            topk_coords = torch.cat([
                topk_coords[..., :2],
                torch.zeros_like(topk_coords[..., 2:4])
            ], dim=-1)

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )

        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict