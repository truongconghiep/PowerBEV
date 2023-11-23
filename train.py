# ------------------------------------------------------------------------
# PowerBEV
# Copyright (c) 2023 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from FIERY (https://github.com/wayveai/fiery)
# Copyright (c) 2021 Wayve Technologies Limited. All Rights Reserved.
# ------------------------------------------------------------------------

import os
import socket
import time

import pytorch_lightning as pl
import torch
from powerbev.config import get_cfg, get_parser
from powerbev.data import prepare_powerbev_dataloaders
from powerbev.trainer import TrainingModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

import torch.nn as nn


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"


def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)

def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    trainloader, valloader = prepare_powerbev_dataloaders(cfg)
    model = TrainingModule(cfg.convert_to_dict())

    if cfg.PRETRAINED.LOAD_WEIGHTS:
        # Load single-image instance segmentation model.
        pretrained_model_weights = torch.load(
            cfg.PRETRAINED.PATH , map_location='cpu'
        )['state_dict']

        model.load_state_dict(pretrained_model_weights, strict=False)
        print(f'Loaded single-image model weights from {cfg.PRETRAINED.PATH}')

    if args.freeze_encoder == True:
        freeze_modules = ['encoder']
        for key, parameter in  model.named_parameters():
            for module_name in freeze_modules:
                if module_name in key:
                    parameter.requires_grad = False

    if args.reset_decoder == True:
        reset_all_weights(model.model.stconv)


    save_dir = os.path.join(
        cfg.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    ) 
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
    checkpoint_callback = ModelCheckpoint(monitor='vpq', save_top_k=5, mode='max')
    trainer = pl.Trainer(
        gpus=cfg.GPUS,
        accelerator='ddp',
        precision=cfg.PRECISION,
        sync_batchnorm=True,
        gradient_clip_val=cfg.GRAD_NORM_CLIP,
        max_epochs=cfg.EPOCHS,
        weights_summary='full',
        logger=tb_logger,
        log_every_n_steps=cfg.LOGGING_INTERVAL,
        plugins=DDPPlugin(find_unused_parameters=True),
        profiler='simple',
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, trainloader, valloader)


if __name__ == "__main__":
    main()