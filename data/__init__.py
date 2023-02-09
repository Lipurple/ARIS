# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>


from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
from data.dataloader import MSDataLoader

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        ##change
        self.loader_testtrain=[]
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                if d in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109','CBSD68','Rain100L','GOPRO_Large']:
                    m = import_module('data.' + 'finetune')
                    datasets.append(getattr(m, 'DIV2K')(args, train=True, name=d))
                elif d == 'attention':
                    m = import_module('data.' + 'attentmap')
                    datasets.append(getattr(m, 'DIV2K')(args, train=True, name=d))
                else:
                    module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                    m = import_module('data.' + module_name.lower())
                    datasets.append(getattr(m, module_name)(args, train=True, name=d))

            self.loader_train = MSDataLoader(
                args,
                datasets[0],
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
            )


        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109','CBSD68','Rain100L','GOPRO_Large']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            elif d == 'DIV2K':
                m = import_module('data.' + 'try')
                testset = getattr(m, 'DIV2K')(args, train=False, name=d)
            elif d == 'finetune':
                m = import_module('data.' + 'finetune_test')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=args.test_batch_size,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )

class Data_video:
    def __init__(self, args):
        print("using data_video")
        self.loader_train = None
        ##change
        self.loader_testtrain=[]
        if not args.test_only:
            datasets = []
            for d in args.data_train:
            ##change
                if d == 'tinyvimeotrain':
                    mtrain = import_module('data.' + 'Tinyvim90ktrain')
                    datasets.append(getattr(mtrain, 'Tinyvimeotrain')(args, train=True,name=d))
                    testtrainset = getattr(mtrain, 'Tinyvimeotrain')(args, train=False,name=d)
                self.loader_testtrain.append(
                dataloader.DataLoader(
                    testtrainset,
                    batch_size=args.test_batch_size,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
            self.loader_train = dataloader.DataLoader(
                datasets[0],
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []
        for d in args.data_test:
            if d == 'tinyvimeotest':
                m = import_module('data.' + 'Tinyvim90ktest')
                testset = getattr(m, 'Tinyvimeotest')(args, train=False,name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=args.test_batch_size,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
