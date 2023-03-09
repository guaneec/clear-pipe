'''Heavily modified from https://github.com/Mikubill/naifu-diffusion/blob/main/lib/callbacks.pyhttps://github.com/Mikubill/naifu-diffusion/blob/main/lib/callbacks.py'''

from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.wandb import WandbLogger
import torch
from pathlib import Path
from typing import *

class SampleCallback(Callback):
    def __init__(self, every_n_steps: int, img_params: List[dict], img_params_shared: Optional[dict] = None):
        self.every_n_steps = every_n_steps
        self.img_params = [{**p, **(img_params_shared or {})} for p in img_params]

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):        
        if self.config is None or self.config.every_n_steps == -1:
            return
        
        if trainer.global_step % self.config.every_n_steps == 0 and trainer.global_step > 0:
            return self.sample(trainer, pl_module)
        
    @torch.inference_mode()
    @rank_zero_only 
    def sample(self, trainer, pl_module):
        save_dir = Path(self.config.save_dir) 
        save_dir.mkdir(parents=True, exist_ok=True)
        images = []
        prompts = []
        for p in self.img_params:
            with torch.random.fork_rng():
                seed = p.pop('seed', -1)
                if seed < 0:
                    torch.seed()
                else:
                    torch.manual_seed(seed)
                images.append(pl_module.sd.txt2img(**p)[0])
                prompts.append(p['prompt'])
        for j, image in enumerate(images):
            image.save(save_dir / f"sample_e{trainer.current_epoch}_s{trainer.global_step}_{j}.png")
        
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.log_image(key="samples", images=images, caption=prompts, step=trainer.global_step)