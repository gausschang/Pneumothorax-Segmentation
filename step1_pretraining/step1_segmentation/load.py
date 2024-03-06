import torch


def load(model, pth_path):
        import copy

        pretrained_dict = torch.load(pth_path)
        pretrained_dict = pretrained_dict['model']
        model_dict = model.state_dict()
        duplicate = copy.deepcopy(model_dict)
        #print(model_dict)

        # 0. filter keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        if len(pretrained_dict)==0:
                raise ValueError(f'model key unmatched! pickup zero pretrained modules')
        print(f'pickup {len(pretrained_dict)} pretrained modules')
        
        # 1. remove tensors with unmatched size
        loaded_modules = 0
        unmatch_list = []
        for i in pretrained_dict:
                if model_dict[i].shape != pretrained_dict[i].shape:
                        unmatch_list.append(i)
                else:
                        loaded_modules += 1
        for i in unmatch_list:
                del pretrained_dict[i]
                
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 

        # 3. load the new state dict
        model.load_state_dict(model_dict)
        #for k in model_dict:
        #        print(k, ' -----  weight delta : ', abs(duplicate[k].to('cpu')-model_dict[k].to('cpu')).sum().item())
        print(f"model summary : {loaded_modules} out of {len(model_dict)} modules were loaded with pretrained weights \n")
        return model

