import argparse
import torch


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument(
            "--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0. use -1 for CPU"
        )
        # SECOND_2排除了裸地变低矮植被的变化类型
        # SECOND_2_ruian 排除了裸地变低矮植被的变化类型,并加上一些瑞安的样本
        self.parser.add_argument("--name", type=str, default="SECOND_ruian_")
        self.parser.add_argument(
            "--dataroot", type=str, default="/mnt/wuxy/change_detection/datasets/all_elements"
        )
        self.parser.add_argument("--dataset", type=str, default="SECOND")
        self.parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="/mnt/wuxy/change_detection/Changedino_dynamic_all_elements/checkpoints/test_SECOND",
            help="models are saved here",
        )
        
        self.parser.add_argument(
            "--save_test", default=True
        )
        self.parser.add_argument(
            "--result_dir", type=str, default="/mnt/wuxy/change_detection/Changedino_dynamic_all_elements/results/SECOND_ruian/test", help="results are saved here"
        )
        self.parser.add_argument(
            "--vis_path", type=str, default="vis", help="results are saved here"
        )
        self.parser.add_argument("--load_pretrain", default=False, help="load pretrained model for testing")
        self.parser.add_argument("--resume", default=False, help="resume training from checkpoint")
        self.parser.add_argument("--use_morph", action='store_true')

        self.parser.add_argument("--phase", type=str, default="train")
        self.parser.add_argument("--test_phase", type=str, default="test")
        self.parser.add_argument("--backbone", type=str, default="mobilenetv2")
        self.parser.add_argument("--fpn", type=str, default="fpn")
        self.parser.add_argument("--fpn_channels", type=int, default=128)
        self.parser.add_argument("--deform_groups", type=int, default=4)
        self.parser.add_argument("--gamma_mode", type=str, default="SE")
        self.parser.add_argument("--beta_mode", type=str, default="contextgatedconv")
        self.parser.add_argument('--n_layers', nargs='+', type=int, default=[1, 1, 1, 1])
        self.parser.add_argument('--extract_ids', nargs='+', type=int, default=[5, 11, 17, 23])
        self.parser.add_argument("--alpha", type=float, default=0.25)
        self.parser.add_argument("--gamma", type=int, default=4, help="gamma for Focal loss")

        self.parser.add_argument("--batch_size", type=int, default=6)
        self.parser.add_argument("--val_batch_size", type=int, default=None)
        self.parser.add_argument("--num_epochs", type=int, default=100)
        self.parser.add_argument("--num_workers", type=int, default=4, help="#threads for loading data")
        self.parser.add_argument("--lr", type=float, default=5e-4)
        self.parser.add_argument("--weight_decay", type=float, default=5e-4)
        self.parser.add_argument("--train_size", type=int, default=512, help="Training input size (width and height). Should be multiple of 16. Default: 512")
        self.parser.add_argument("--accelerator", type=str, default="gpu")
        self.parser.add_argument("--devices", type=str, default="1")
        self.parser.add_argument("--strategy", type=str, default="auto")
        self.parser.add_argument("--precision", type=str, default="16")
        self.parser.add_argument("--save_ckpt_mode", type=str, default="state_dict")

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(",")
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print("------------ Options -------------")
        for k, v in sorted(args.items()):
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")

        return self.opt
