import logging

import numpy as np
import torch
import tqdm
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from molutils import MolAnalyzer
from utils import ModelSaver


class Trainer:
    def __init__(self, model, device, config, logdir=None):

        self._iteration = 1

        self._device = device
        self.config = config

        self.model = model.to(device)
        self.train_losses = {}

        logging.getLogger("main").info(f"Model size: {sum(p.numel() for p in self.model.parameters())}")

        self.train_set, self.test_set, self.train_loader, self.test_loader = model.get_data_loader()

        # self.opt = SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.opt = Adam(self.model.parameters(), lr=config["lr"])

        optimizer_path = config.get("optimizer_cp_path")
        if optimizer_path:
            self.opt.load_state_dict(torch.load(optimizer_path))

        self.lr_scheduler = ReduceLROnPlateau(optimizer=self.opt)

        self.max_grad_norm = config["max_grad_norm"]

        if config["log_to_tensorboard"]:

            self.writer = SummaryWriter(log_dir=logdir)
            logging.getLogger("main").info(f"writing tensorboard log to {self.writer.log_dir}")

            self.modelsaver = ModelSaver(
                model=self.model,
                optimizer=self.opt,
                save_dir=self.writer.log_dir,
            )

            with open(f"{self.writer.log_dir}/run_config.yaml", "w") as f:
                yaml.dump(config, f)

        self.mol_analyzer = MolAnalyzer()

    def train(self, e):

        logging.getLogger("main").info(f"Number of training samples: {len(self.train_set)}")
        logging.getLogger("main").info(f"Number of test samples: {len(self.test_set)}")

        for e in range(e):
            early_stop = self.train_loop(e)
            if self.config["early_stopping"] and early_stop:
                logging.getLogger("main").info("Loss did not increase over a period of time. Stopping training...")
                return

    def train_loop(self, e):
        for batched_info in tqdm.tqdm(self.train_loader, desc=f"epoch {e}"):

            self.train_step(batched_info)

        early_stop = self.evaluate(sample_molecules=self.config["sample"])
        return early_stop

    def train_step(self, training_batch):

        self.model.train()

        losses = self.pass_through_model(
            batched_info=training_batch,
        )

        self.opt.zero_grad()
        losses["loss"].backward()
        gradnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.opt.step()

        self._log_loss(losses, self._iteration)

        self._iteration += 1

    def evaluate(
        self, num_samples=100, eval_on_test_set=True, sample_molecules=True, output_dir=None, atoms_to_complete=None
    ):

        self.model.eval()

        early_stop = False

        # evaluate on the test set
        if eval_on_test_set:
            early_stop = self.evaluate_on_test_set()

        if not sample_molecules:
            return early_stop

        # evaluate by generating new molecules
        sampled_info = []

        with tqdm.tqdm(total=num_samples, desc="Cooking some new molecules: ") as pbar:
            while len(sampled_info) < num_samples:
                to_sample = min(num_samples - len(sampled_info), self.config["batch_size"])
                sampled_molecules = self.model.sample(num_samples=to_sample, device=self._device)
                [sampled_info.append(i) for i in sampled_molecules]
                pbar.update(to_sample)

        # Calculate the metrics for the sampled molecules
        metrics, nr_valid, nr_wrong_valence, nr_not_connected = self.mol_analyzer.analyze_mol_tensor(
            sampled_info[:num_samples], output_dir=output_dir
        )
        print(nr_valid)
        print(nr_wrong_valence)
        print(nr_not_connected)
        self._log(tag="gen_mols/nr_valid", value=nr_valid, step=self._iteration)
        self._log(tag="gen_mols/nr_not_connected", value=nr_not_connected, step=self._iteration)
        self._log(tag="gen_mols/nr_wrong_valence", value=nr_wrong_valence, step=self._iteration)

        return early_stop

    def evaluate_on_test_set(self):

        test_losses = []

        for batched_info in tqdm.tqdm(self.test_loader, desc="testing:"):

            losses = self.pass_through_model(batched_info=batched_info, train=False)
            test_losses.append(losses)

        test_losses = {k: torch.mean(torch.tensor([l[k] for l in test_losses])).item() for k in test_losses[0]}
        for loss_key in test_losses:
            self._log(tag=f"loss_test/{loss_key}", value=test_losses[loss_key], step=self._iteration)

        if self.config["lr_scheduler"]:
            self.lr_scheduler.step(test_losses["loss"])

        early_stop = self.modelsaver.step(cur_loss=test_losses["loss"])

        for param_group in self.opt.param_groups:
            lr = param_group["lr"]
        self._log(tag=f"learning_rate", value=lr, step=self._iteration)

        return early_stop

    def pass_through_model(self, batched_info, train=True):
        """
        forward pass through the model and calculation of loss

        """
        if train:
            predictions = self.model(batched_info["x"])
        else:
            with torch.no_grad():
                predictions = self.model(batched_info["x"])

        losses = self.model.calculate_loss(predictions, batched_info)
        return losses

    def _log_loss(self, loss_dict, step):
        if not self.config["log_to_tensorboard"]:
            return

        for loss in loss_dict:
            try:
                loss_list = self.train_losses[loss]
            except KeyError:
                loss_list = []
                self.train_losses[loss] = loss_list

            loss_list.append(loss_dict["loss"].item())

            running_mean_interval = self.config["running_mean_interval"]
            if len(loss_list) == running_mean_interval:
                mean = np.mean(loss_list)
                self._log(tag=f"loss_train/{loss}", value=mean, step=step)
                loss_list.clear()

    def _log(self, tag, value, step):
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)
