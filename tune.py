import pytorch_lightning as pl
from jsonargparse import Namespace
from typing import *
import torch
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
from transformers import CLIPTextModel, CLIPTokenizer
from clear_pipe.sd import StableDiffusion
from clear_pipe.data import ClearDataModule
import re
import wandb

class HiddenModule:
    def __init__(self, module):
        self.module = module


class DeltaLinear(torch.nn.Module):
    def __init__(self, orig: torch.nn.Linear) -> None:
        super().__init__()
        if orig.bias is not None:
            self.bias = torch.nn.Parameter(torch.zeros_like(orig.bias, dtype=float))
        self.weight = torch.nn.Parameter(torch.zeros_like(orig.weight, dtype=float))
        self.orig = HiddenModule(orig)

    def forward(self, x):
        import torch.nn.functional as F

        orig = self.orig.module
        weight = orig.weight + self.weight.to(orig.weight)
        bias = None if orig.bias is None else orig.bias + self.bias.to(orig.bias)
        return F.linear(x, weight, bias)


class StableDiffusionTuner(pl.LightningModule):
    def __init__(
        self,
        *,
        sd: StableDiffusion,
        embedding_lr: float = 1.0,
        model_lr: float = 1.0,
        model_unfrozen_regex: str = r"2.to_[kv]",
        fixed_image_log_params: Optional[dict],
        log_random_image: bool = True,
        log_image_every_nsteps: int = 0,
        export_every_nsteps: int = 0,
        optimizer_params: Optional[dict],
    ):
        super().__init__()

        self.embeds = sd.db
        self.model_weights = torch.nn.ModuleDict()
        for k, v in sd.named_modules():
            if model_unfrozen_regex and re.search(model_unfrozen_regex, k):
                module_path, _dot, kp = k.rpartition(".")
                assert type(v) == torch.nn.Linear
                parent = sd.get_submodule(module_path)
                self.model_weights[k.replace(".", ">")] = DeltaLinear(v)
                setattr(parent, kp, self.model_weights[k.replace(".", ">")])

        self.log_random_image = log_random_image
        self.fixed_image_log_params = fixed_image_log_params or {}
        self.embedding_lr = embedding_lr
        self.model_lr = model_lr
        self.log_image_every_nsteps = log_image_every_nsteps
        self.export_every_nsteps = export_every_nsteps
        self.sd = HiddenModule(sd)
        self.optimizer_params = optimizer_params
        wandb.init(project="tuning")

    def _gen_image(self, **kwargs):
        print("prompt:", kwargs["prompt"])
        with torch.random.fork_rng():
            image = self.sd.module(**kwargs)[0]
        return image

    def _log_image(self, prompt: str):
        from torch.utils.tensorboard.writer import SummaryWriter

        tensorboard: SummaryWriter = self.logger.experiment
        from torchvision.transforms.functional import to_tensor

        if self.log_random_image:
            rand_image = self._gen_image(prompt=prompt)
            tensorboard.add_image(
                "imgs/random", to_tensor(rand_image), global_step=self.global_step
            )
        if self.fixed_image_log_params:
            fixed_image = self._gen_image(**self.fixed_image_log_params)
            tensorboard.add_image(
                "imgs/fixed", to_tensor(fixed_image), global_step=self.global_step
            )
        pl.LightningModule().named_modules()

    def export(
        self, path: str, weight_dtype: torch.dtype = torch.float16, top_sum: float = 1.0
    ):
        from safetensors.torch import save_file
        import json

        tensors = {}
        embeddings = []
        weights = {}
        for k, v in self.embeds.items():
            tensors[f"embeddings/{k}"] = v
            embeddings.append(k)
        for module_path, delta_module in self.model_weights.items():
            for k, v in delta_module.named_parameters():
                param_path = f"{module_path.replace('>', '.')}.{k}"
                if top_sum == 1:  # no factorization
                    tensors[f"weights/{param_path}"] = v.to(weight_dtype)
                    weights[param_path] = "delta"
                else:
                    (
                        tensors[f"weights/{param_path}.US"],
                        tensors[f"weights/{param_path}.Vh"],
                    ) = map(
                        lambda a: a.to(weight_dtype).contiguous(), decompose(v, top_sum)
                    )
                    weights[param_path] = "delta_factors"
        metadata = dict(version="0.1.0", embeddings=embeddings, weights=weights)
        save_file(tensors, path, {"tuner": json.dumps(metadata)})

    def training_step(self, batch, batch_idx):
        x = batch.latent
        c = self.sd.module.patched_clip(batch.cond_text)
        b = len(x)
        t = torch.randint(0, 1000, (b,)).long()
        model_pred = self.sd.module.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = shared.sd_model(x, c)[0]
        self.log("loss/train", loss.item())
        if (
            self.log_image_every_nsteps
            and (self.global_step + 1) % self.log_image_every_nsteps == 0
        ):
            self._log_image(batch.cond_text[0])

        if (
            self.export_every_nsteps
            and (self.global_step + 1) % self.export_every_nsteps == 0
        ):
            path = os.path.join(
                self._trainer.log_dir, f"{self.global_step}.tuner.safetensors"
            )
            print(path)
            self.export(path, top_sum=0.5)

        return loss

    def configure_optimizers(self):
        param_groups = []
        if self.embeds:
            param_groups.append(
                dict(params=list(self.embeds.values()), lr=self.embedding_lr)
            )
        if self.model_weights:
            param_groups.append(
                dict(
                    params=list(self.model_weights.parameters()),
                    lr=self.model_lr,
                    eps=1e-5,
                )
            )
        return self.optimzer(param_groups)


class ClearCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_class_arguments(StableDiffusion, "sd")
        parser.link_arguments("sd", "model.sd", apply_on="instantiate")
        parser.link_arguments("sd", "data.sd", apply_on="instantiate")


if __name__ == "__main__":
    ClearCLI(
        StableDiffusionTuner, ClearDataModule, trainer_defaults={"log_every_n_steps": 1}
    )
