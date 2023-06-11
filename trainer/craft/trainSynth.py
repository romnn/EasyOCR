# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time
import yaml
import multiprocessing as mp
from pprint import pprint
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import wandb

from config.load_config import load_yaml, DotDict
from data.dataset import SynthTextDataSet
from loss.mseloss import Maploss_v2, Maploss_v3
from model.craft import CRAFT
from metrics.eval_det_iou import DetectionIoUEvaluator
from eval import main_eval
from utils.util import copyStateDict, save_parser
import warnings

# turn matching warnings into exceptions
warnings.filterwarnings(action="ignore", category=ResourceWarning, module=r".*wandb")
warnings.filterwarnings(action="ignore", category=ResourceWarning, module=r"tempfile")


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"


class Trainer(object):
    def __init__(self, config, gpu):
        self.config = config
        self.gpu = gpu
        self.mode = None
        self.trn_loader, self.trn_sampler = self.get_trn_loader()
        self.net_param = self.get_load_param(gpu)

    def get_trn_loader(self):
        dataset = SynthTextDataSet(
            output_size=self.config.train.data.output_size,
            data_dir=self.config.data_dir.synthtext,
            saved_gt_dir=None,
            mean=self.config.train.data.mean,
            variance=self.config.train.data.variance,
            gauss_init_size=self.config.train.data.gauss_init_size,
            gauss_sigma=self.config.train.data.gauss_sigma,
            enlarge_region=self.config.train.data.enlarge_region,
            enlarge_affinity=self.config.train.data.enlarge_affinity,
            aug=self.config.train.data.syn_aug,
            vis_test_dir=self.config.vis_test_dir,
            vis_opt=self.config.train.data.vis_opt,
            sample=self.config.train.data.syn_sample,
        )

        trn_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        trn_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            sampler=trn_sampler,
            drop_last=True,
            pin_memory=True,
        )
        return trn_loader, trn_sampler

    def get_load_param(self, gpu):
        if self.config.train.ckpt_path is not None:
            map_location = {"cuda:%d" % 0: "cuda:%d" % gpu}
            param = torch.load(self.config.train.ckpt_path, map_location=map_location)
        else:
            param = None
        return param

    def adjust_learning_rate(self, optimizer, gamma, step, lr):
        lr = lr * (gamma**step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return param_group["lr"]

    def get_loss(self):
        if self.config.train.loss == 2:
            criterion = Maploss_v2()
        elif self.config.train.loss == 3:
            criterion = Maploss_v3()
        else:
            raise Exception("Undefined loss")
        return criterion

    def iou_eval(self, dataset_name, train_step, save_param_path, buffer, model):
        test_config = DotDict(self.config.test[dataset_name])

        val_result_dir = Path(self.config.results_dir) / f"{dataset_name}_iou/{train_step}"

        evaluator = DetectionIoUEvaluator()

        metrics = main_eval(
            model_path=save_param_path,
            backbone=self.config.train.backbone,
            config=test_config,
            dataset_name=dataset_name,
            evaluator=evaluator,
            result_dir=val_result_dir,
            # buffer,
            model=model,
            mode=self.mode,
        )
        pprint(metrics)
        if self.gpu == 0 and self.config.wandb_opt:
            wandb.log(
                {
                    "{} IoU Recall".format(dataset_name): np.round(metrics["recall"], 3),
                    "{} IoU Precision".format(dataset_name): np.round(metrics["precision"], 3),
                    "{} IoU F1-score".format(dataset_name): np.round(metrics["hmean"], 3),
                }
            )

    def train(self, buffer_dict):
        torch.cuda.set_device(self.gpu)

        # DATASET
        trn_loader = self.trn_loader

        # MODEL
        if self.config.train.backbone == "vgg":
            craft_model = CRAFT(pretrained=True, amp=self.config.train.amp)
        else:
            raise ValueError("Undefined architecture")

        if self.config.train.ckpt_path is not None:
            craft_model.load_state_dict(copyStateDict(self.net_param["craft"]))
        craft = nn.SyncBatchNorm.convert_sync_batchnorm(craft_model)
        craft = craft.cuda()

        summary(
            craft_model,
            input_size=(
                self.config.train.batch_size,
                3,  # 3 (RGB) color channels
                self.config.train.data.output_size,  # width
                self.config.train.data.output_size,  # height
            ),
        )

        craft = torch.nn.parallel.DistributedDataParallel(craft, device_ids=[self.gpu])

        torch.backends.cudnn.benchmark = True

        # OPTIMIZER

        optimizer = optim.Adam(
            craft.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )

        if self.config.train.ckpt_path is not None and self.config.train.st_iter != 0:
            optimizer.load_state_dict(copyStateDict(self.net_param["optimizer"]))
            self.config.train.st_iter = self.net_param["optimizer"]["state"][0]["step"]
            self.config.train.lr = self.net_param["optimizer"]["param_groups"][0]["lr"]

        # LOSS
        if self.config.train.amp:
            # mixed precision
            scaler = torch.cuda.amp.GradScaler()

            # load model
            if self.config.train.ckpt_path is not None and self.config.train.st_iter != 0:
                scaler.load_state_dict(copyStateDict(self.net_param["scaler"]))
        else:
            scaler = None

        criterion = self.get_loss()

        # TRAIN
        train_step = self.config.train.st_iter
        whole_training_step = self.config.train.end_iter
        update_lr_rate_step = 0
        training_lr = self.config.train.lr
        loss_value = 0
        batch_time = 0
        epoch = 0
        start_time = time.time()

        while train_step < whole_training_step:
            self.trn_sampler.set_epoch(train_step)
            for (
                index,
                (
                    image,
                    region_image,
                    affinity_image,
                    confidence_mask,
                ),
            ) in enumerate(trn_loader):
                craft.train()
                if train_step > 0 and train_step % self.config.train.lr_decay == 0:
                    update_lr_rate_step += 1
                    training_lr = self.adjust_learning_rate(
                        optimizer,
                        self.config.train.gamma,
                        update_lr_rate_step,
                        self.config.train.lr,
                    )

                images = image.cuda(non_blocking=True)
                region_image_label = region_image.cuda(non_blocking=True)
                affinity_image_label = affinity_image.cuda(non_blocking=True)
                confidence_mask_label = confidence_mask.cuda(non_blocking=True)

                if self.config.train.amp:
                    # autocast back to f32
                    with torch.cuda.amp.autocast():
                        output, _ = craft(images)
                        out1 = output[:, :, :, 0]
                        out2 = output[:, :, :, 1]

                        loss = criterion(
                            region_image_label,
                            affinity_image_label,
                            out1,
                            out2,
                            confidence_mask_label,
                            self.config.train.neg_rto,
                            self.config.train.n_min_neg,
                        )

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    output, _ = craft(images)
                    out1 = output[:, :, :, 0]
                    out2 = output[:, :, :, 1]
                    loss = criterion(
                        region_image_label,
                        affinity_image_label,
                        out1,
                        out2,
                        confidence_mask_label,
                        self.config.train.neg_rto,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                end_time = time.time()
                loss_value += loss.item()
                batch_time += end_time - start_time

                if train_step > 0 and train_step % 5 == 0 and self.gpu == 0:
                    mean_loss = loss_value / 5
                    loss_value = 0
                    avg_batch_time = batch_time / 5
                    batch_time = 0

                    print(
                        "{}, training_step: {}|{}, learning rate: {:.8f}, "
                        "training_loss: {:.5f}, avg_batch_time: {:.5f}".format(
                            time.strftime("%Y-%m-%d:%H:%M:%S", time.localtime(time.time())),
                            train_step,
                            whole_training_step,
                            training_lr,
                            mean_loss,
                            avg_batch_time,
                        )
                    )
                    if self.gpu == 0 and self.config.wandb_opt:
                        wandb.log({"train_step": train_step, "mean_loss": mean_loss})

                if train_step % self.config.train.eval_interval == 0 and train_step != 0:
                    # initialize all buffer value with zero
                    if self.gpu == 0:
                        for buffer in buffer_dict.values():
                            for i in range(len(buffer)):
                                buffer[i] = None

                    print("Saving state, index:", train_step)
                    save_param_dic = {
                        "iter": train_step,
                        "craft": craft.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_param_path = self.config.results_dir + "/CRAFT_clr_" + repr(train_step) + ".pth"

                    if self.config.train.amp:
                        save_param_dic["scaler"] = scaler.state_dict()
                        save_param_path = self.config.results_dir + "/CRAFT_clr_amp_" + repr(train_step) + ".pth"

                    if self.gpu == 0:
                        torch.save(save_param_dic, save_param_path)

                    # validation
                    self.iou_eval(
                        "icdar2013",
                        train_step,
                        save_param_path,
                        buffer_dict["icdar2013"],
                        craft,
                    )

                train_step += 1
                if train_step >= whole_training_step:
                    break
            epoch += 1

        # save last model
        if self.gpu == 0:
            save_param_dic = {
                "iter": train_step,
                "craft": craft.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_param_path = self.config.results_dir + "/CRAFT_clr_" + repr(train_step) + ".pth"

            if self.config.train.amp:
                save_param_dic["scaler"] = scaler.state_dict()
                save_param_path = self.config.results_dir + "/CRAFT_clr_amp_" + repr(train_step) + ".pth"
            torch.save(save_param_dic, save_param_path)


# def get_num_params(model, learnable=False):
#     pp = 0
#     for p in list(model.parameters()):
#         if learnable and not p.requires_grad:
#             continue
#         nn = 1
#         for s in list(p.size()):
#             nn *= s
#         pp += nn
#     return pp


def main():
    parser = argparse.ArgumentParser(description="CRAFT SynthText Train")
    parser.add_argument(
        "--yaml",
        "--yaml_file_name",
        default="syn_train",
        type=str,
        help="Load configuration",
    )
    parser.add_argument("--port", "--use ddp port", default="2646", type=str, help="Load configuration")

    args = parser.parse_args()

    # load configure
    exp_name = args.yaml
    config = load_yaml(args.yaml)

    print("-" * 20 + " Options " + "-" * 20)
    print(yaml.dump(config))
    print("-" * 40)

    # CRAFT(pretrained=False, freeze=False, amp=False)().summary()
    # num_params = get_num_params(CRAFT(pretrained=False, freeze=False, amp=False))
    # print(
    #     "CRAFT has %d parameters (%d MB using f32)" % (num_params, num_params * 4 / (1024**2)),
    # )

    # Make result_dir
    res_dir = os.path.join(config["results_dir"], args.yaml)
    config["results_dir"] = res_dir
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Duplicate yaml file to result_dir
    shutil.copy("config/" + args.yaml + ".yaml", os.path.join(res_dir, args.yaml) + ".yaml")

    ngpus_per_node = torch.cuda.device_count()
    print(f"Total device num : {ngpus_per_node}")

    manager = mp.Manager()
    buffer1 = manager.list([None] * config["test"]["icdar2013"]["test_set_size"])
    buffer_dict = {"icdar2013": buffer1}
    torch.multiprocessing.spawn(
        main_worker,
        nprocs=ngpus_per_node,
        args=(
            args.port,
            ngpus_per_node,
            config,
            buffer_dict,
            exp_name,
        ),
    )
    print("done")


def main_worker(gpu, port, ngpus_per_node, config, buffer_dict, exp_name):
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:" + port,
        world_size=ngpus_per_node,
        rank=gpu,
    )

    # Apply config to wandb
    if gpu == 0 and config["wandb_opt"]:
        wandb.init(project="craft-stage1", entity="gmuffiness", name=exp_name)
        wandb.config.update(config)

    batch_size = int(config["train"]["batch_size"] / ngpus_per_node)
    config["train"]["batch_size"] = batch_size
    config = DotDict(config)

    # Start train
    trainer = Trainer(config, gpu)
    trainer.train(buffer_dict)

    if gpu == 0 and config["wandb_opt"]:
        wandb.finish()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
