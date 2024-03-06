import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from tinyvit_cls import TinyViT_cls
from tinyvit import TinyViT
from load import *

class get_inferrer_cls(nn.Module):
    def __init__(self, 
        config={'encoder_name':None, 
                'method': None,
                'aux_params':None,
                'weight_list':[]
                },
        device='cuda'
                ):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.models = self.load_models()

    def load_models(self):
        config = self.config
        device = self.device

        models = []
        for w in config['weight_list']:
            if config['encoder_name'] is not None:
                model = getattr(smp, config['method'])(
                            encoder_name=config['encoder_name'],      # choose encoder
                            encoder_weights=None,           # skip download weight
                            in_channels=3,
                            classes=2,
                            aux_params=config['aux_params']
                        )
            else:
                model = TinyViT_cls(
                            img_size=512,
                            num_classes=2,
                            embed_dims=[96, 192, 384, 576],
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 18],
                            window_sizes=[16, 16, 32, 16],
                            drop_path_rate=0.1,
                        )

            model = load(model, w)
            model.to(device)
            model.eval()
            models.append(model)
        return models

    def forward(self, x: torch.Tensor):
        seg = []
        logit = []
        for m in self.models:
            with torch.no_grad():
                seg1, logit1 = m(x)
                seg1, logit1 = F.softmax(seg1, dim=1), torch.sigmoid(logit1)

                if len(self.config['weight_list'])>1:
                    seg.append(seg1)
                    logit.append(logit1)

                else:
                    seg = seg1
                    logit = logit1

        if len(self.config['weight_list'])>1:
            seg = torch.stack(seg, dim=0) # models,b,c,h,w
            seg = torch.mean(seg, dim=0) #  b,c,h,w

            logit = torch.stack(logit, dim=0)
            logit = torch.mean(logit, dim=0)

        return seg[:, 1,...].unsqueeze(dim=1).to('cpu'), logit.to('cpu')


class get_inferrer_seg(nn.Module):
    def __init__(self, 
        config={'encoder_name':None, 
                'method': None,
                'weight_list':[]
                },
        device='cuda'
                ):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.models = self.load_models()

    def load_models(self):
        config = self.config
        device = self.device

        models = []
        for w in config['weight_list']:
            if config['encoder_name'] != None:
                model = getattr(smp, config['method'])(
                            encoder_name=config['encoder_name'],      # choose encoder
                            encoder_weights=None,           # skip download weight
                            in_channels=3,
                            classes=2,
                        )
            else:
                model = TinyViT(
                            img_size=512,
                            num_classes=2,
                            embed_dims=[96, 192, 384, 576],
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 18],
                            window_sizes=[16, 16, 32, 16],
                            drop_path_rate=0.1,
                        )

            model.load_state_dict(torch.load(w))
            model.to(device)
            model.eval()
            models.append(model)
        return models

    def forward(self, x: torch.Tensor):
        seg = []
        for m in self.models:
            with torch.no_grad():
                seg1 = m(x)   # b,c,h,w
                seg1 = F.softmax(seg1, dim=1)

                if len(self.config['weight_list'])>1:
                    seg.append(seg1)
                else:
                    seg = seg1

        if len(self.config['weight_list'])>1:
            seg = torch.stack(seg, dim=0) # models,b,c,h,w
            seg = torch.mean(seg, dim=0) #  b,c,h,w

        return seg[:, 1,...].unsqueeze(dim=1).to('cpu')