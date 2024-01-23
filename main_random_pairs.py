import argparse
import torch
from torch.nn.functional import kl_div
import pandas as pd
from tqdm import tqdm
import logging
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src.prompt.machine_prompt import RandomPairPromptSearch
from src.prompt.utils import parse_paraphrases
from src.data.lama_dataset import LAMAset
from src.utils.init_utils import init_device, init_random_seed
from src.model.causal_lm import CausalLanguageModel
from src.data.dataset_loader import batchify


def parse_args():
    parser = argparse.ArgumentParser(description='AMAP')

    # Data selection
    parser.add_argument('--model_name', type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cuda', help='Which computation device: cuda or mps')
    parser.add_argument('--output_dir', type=str, default='./amap', help='the output directory to store prediction results')
    parser.add_argument('--fp16', action='store_true', help='use half precision')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--n_iterations_max', type=int, default=100)
    parser.add_argument('--n_population', type=int, default=50)
    parser.add_argument('--num_candidates', type=int, default=5)
    parser.add_argument('--template_len', type=int, default=5)
    parser.add_argument('--n_pairs', type=int, default=1)
    parser.add_argument('--n_searches', type=int, default=1)



    args = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":

    args = parse_args()
    
    random_seed = init_random_seed(args.seed)
    init_device(args.device)

    # load LM
    print("Loading model...")
    model_name = args.model_name
    model = CausalLanguageModel(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        fast_tkn=True if not ('opt' in model_name) else False, #because of a bug in OPT
        fp16=args.fp16,
        padding_side='left')
    model_name_parse = model_name.split('/')[-1]

    # output file
    savepath = os.path.join(args.output,f'random-pairs-prompt-search_{model_name_parse}_p{args.n_pairs}_t{args.template_len}_{random_seed}.tsv')
    with open(savepath, 'w') as f_out:
        savelines = f'source\ttarget\tcpt_iteration\tbest_prompt\ttemplate\tscore\n'
        f_out.write(savelines)

    # initialise the search algo
    promptSearch = RandomPairPromptSearch(model, args.n_population, args.num_candidates, n_rounds=2, verbose=True)


    for i in range(args.n_searches):
        target_pairs = [(model.tokenizer.decode(random.randint(0, model.get_vocab_size()-1)), 
                        model.tokenizer.decode(random.randint(0, model.get_vocab_size()-1))) for _ in range (args.n_pairs)]
        print(target_pairs)

        all_population_template, cpt_iteration = promptSearch.train(template_len=args.template_len, target_pairs=target_pairs, n_iterations_max=args.n_iterations_max, batch_size=args.batch_size, savepath=None)
        best_prompt = all_population_template[0]
        print(best_prompt)
        #write into file
        if savepath is not None:
            with open(savepath, 'a') as f_out:
                savelines = f'{target_pairs[0]}\t{target_pairs[1]}\t{cpt_iteration}\t{best_prompt[2]}\t[START-TEMPLATE]{best_prompt[0]}[END-TEMPLATE]\t{best_prompt[1]:.2f}\n'
                f_out.write(savelines)