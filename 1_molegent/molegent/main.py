import sys
sys.path.append(".")

from models.spatial_transformer import SpatialTransformer

from trainer import Trainer
import argparse
import torch
import logging 
import yaml 
import torch.nn as nn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", required=True, help="The path to the config file in yaml format. ")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--model_type", choices=["spatial", "graph"], default="graph")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--cp_path", help="The path to a model checkpoint. ")
    parser.add_argument("--logdir", type=str, help="Tensorboard and model checkpoints will be saved here. ")
    parser.add_argument("--ngpus", type=int, help="number of gpus")

    return parser.parse_args()


def parse_config(configfile: str):
    with open(configfile, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main():
    args = parse_arguments()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("main").info(f"cuda available: {torch.cuda.is_available()}")

    config = parse_config(args.config_file)

    transformer = None

    if args.model_type == "spatial":
        transformer = SpatialTransformer(config=config, device=args.device)
    elif args.model_type == "graph":
        from molegent.models.graph_transformer import GraphTransformer 
        transformer = GraphTransformer(config=config, device=args.device)
    
    # transformer = nn.DataParallel(transformer, device_ids=list(range(args.ngpus))).cuda()

    trainer = Trainer(transformer, config=config, device=args.device, logdir=args.logdir)

    if args.train:
        if args.cp_path:
            #transformer.load_state_dict(torch.load(args.cp_path, map_location=args.device))
            transformer.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.cp_path, map_location=args.device).items()})
            trainer.train(config["num_epochs"])#I add
        if args.pretrain:
            trainer.train(config["num_epochs"])#I add
            #transformer.load_state_dict(torch.load(args.cp_path, map_location=args.device))
        #trainer.train(config["num_epochs"])
    else:
        assert args.cp_path, "Interference requires a path to a model checkpoint"
        transformer.load_state_dict(torch.load(args.cp_path, map_location=args.device))
        trainer.evaluate(
            num_samples=config["num_samples"],
            sample_molecules=config["sample"],
            eval_on_test_set=False,
            output_dir=config["inference_output_dir"],
        )


if __name__ == "__main__":
    main()
