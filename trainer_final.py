import utility
import random
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import os
import math
from decimal import Decimal
from data import common
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from PIL import Image
import numpy as np
from losses import SupConLoss
from loss import vgg

def chw_to_pillow2(x):
    normalized = x.mul(255 / 255)
    tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
    return Image.fromarray(tensor_cpu.numpy().astype(np.uint8))

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.scaleh = args.scaleh
        self.save_dir = args.save
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.save_dir+"/results-DIV2K"):
            os.makedirs(self.save_dir+"/results-DIV2K")
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.loader_testtrain=loader.loader_testtrain
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        self.conloss = SupConLoss(temperature=0.07)
        self.ifcon = False
        self.inter = False
        self.bicubic = False
        self.vggloss = vgg.VGG(conv_index='22',rgb_range=255).cuda()
        self.ifpercp = False
        self.ema = EMA(self.model, 0.95)
        self.ifema = False
        self.scalenew = args.scalenew
    
    def test2(self):
        
        if self.ifema:
            self.model = self.ema.apply_shadow(self.model)
        
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    b, c, h, w = lr.shape
                    if h < 48 and w >= 48:
                        lr = img_argument(lr, 'h')
                    elif h >= 48 and w < 48:
                        lr = img_argument(lr, 'w')
                    elif h <48 and w < 48:
                        lr = img_argument(lr, 'hw')
                    else:
                        lr = lr
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    
                    if h < 48 and w >= 48:
                        sr = sr[:,:,:int(h*scale),:]
                    elif h >= 48 and w < 48:
                        sr = sr[:,:,:,:int(w*scale)]
                    elif h <48 and w < 48:
                        sr = sr[:,:,:int(h*scale),:int(w*scale)]
                    else:
                        sr = sr
                    
                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, d.dataset.name, scale, self.args.rgb_range
                        )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        print('test:',self.args.test_only)
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            print('epoch:',epoch)
            return epoch >= self.args.epochs
    
    def getpatch(self, sr, scale_idx):
        s = int(self.scalenew[scale_idx])
        c, h, w = sr.shape[-3:]
        sr_list = []
        for i in range(s):
            for j in range(s):
                sr_list.append(sr[:,:,i*48:(i+1)*48,j*48:(j+1)*48])
        sr_new = torch.stack(sr_list, dim=0).view(-1, c, 48, 48)
        return sr_new
    
    def returnshape(self, srs, scale_idx, crop_idx):
        s = int(self.scalenew[scale_idx])
        cs = int(self.scalenew[crop_idx])
        b, c, h, w = srs.shape
        new_sr = torch.zeros((int(b/(s*s)), c, h*s, w*s)).cuda()
        for i in range(s):
            for j in range(s):
                new_sr[:,:,i*48*cs:(i+1)*48*cs,j*48*cs:(j+1)*48*cs] = srs[(i*s+j)*int(b/(s*s)):(i*s+j+1)*int(b/(s*s)),:,:,:]
        return new_sr
        
class EMA():
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register(model)

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
        return model

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        return model
    
def img_argument(img, type):
    if type=='h':
        img1 = img.cpu().numpy()
        img2 = img1[:,:,::-1,:]
        img2 = torch.from_numpy(np.ascontiguousarray(img2)).cuda()
        img_new = torch.cat((img, img2), dim=2)
        return img_new
    elif type=='w':
        img1 = img.cpu().numpy()
        img2 = img1[:,:,:,::-1]
        img2 = torch.from_numpy(np.ascontiguousarray(img2)).cuda()
        img_new = torch.cat((img, img2), dim=3)
        return img_new
    elif type == 'hw':
        img1 = img.cpu().numpy()
        img2 = img1[:,:,::-1,:]
        img2 = torch.from_numpy(np.ascontiguousarray(img2)).cuda()
        img_new1 = torch.cat((img, img2), dim=2)
        img3 = img1[:,:,:,::-1]
        img4 = img3[:,:,::-1,:]
        img3 = torch.from_numpy(np.ascontiguousarray(img3)).cuda()
        img4 = torch.from_numpy(np.ascontiguousarray(img4)).cuda()
        img_new2 = torch.cat((img3, img4), dim=2)
        img_new = torch.cat((img_new1, img_new2), dim=3)
        return img_new