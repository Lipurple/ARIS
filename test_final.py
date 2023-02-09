from tensorboard_logger import Logger
from option import args
import torch
import utility
import data
import loss
from trainer_final import Trainer
import warnings
warnings.filterwarnings('ignore')
import model
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
import os
scale = 0

def main():
    global model
    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        if args.pretrain != "":
            state_dict = torch.load(args.pretrain)
            model_dict = _model.model.state_dict()
            total = 0
            for (k,v) in state_dict.items():
                if k in model_dict.keys():
                    total += 1
                    model_dict[k] = v
            _model.model.load_state_dict(model_dict,strict = True)
            
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        print('test')
        t.test2()
            
if __name__ == '__main__':
    main()