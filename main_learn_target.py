import argparse
import torch
import pandas as pd
from tqdm import tqdm
import logging
import random
import os
import json

from src.prompt.machine_prompt import DiscreteGradientPromptSearch, OneTokenGradientPromptSearch
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
    parser.add_argument('--fp16', action='store_true', help='use half precision')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--n_iterations_max', type=int, default=100)
    parser.add_argument('--n_population', type=int, default=50)
    parser.add_argument('--num_candidates', type=int, default=5)
    parser.add_argument('--relation', type=str, default='all')
    parser.add_argument('--mode', type=str, default='hot')

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

    #load target
    print("Loading target...")
    datapath = "/home/ckervadec/SYSIF/pythia_1.3b_filter_10-10-10ngram_wiki_20000"
    target_dataset = LAMAset(datapath, portion=1, target=True)

    print("Starting!")


    relations = target_dataset.get_relations(set='train') if args.relation=='all' else [args.relation,]
    # relations = ['T1',]# 'T1', 'T4', 'T7']
    # initialise the algo
    autoprompt = DiscreteGradientPromptSearch(model, args.n_population, args.num_candidates, n_rounds=1, verbose=True, n_tkn_generated=10, metric='bleu')

    for relation in relations: # in the future, run all relation in parallel in different scrips
        # initial_template = paraphrases[relation]
        print(f"Relation {relation}")
        """
        dataset is a list of tuple [(X,Y), ...]
        where X is used to fill in the template and Y is the expected next token.
        """

        random_template=['[X]'+model.tokenizer.decode([random.randint(0, model.get_vocab_size()-1) for _ in range(10)])+' [Y]' for __ in range(10)]
        # load human template
        with open(os.path.join(datapath, relation+'.json')) as f:
            human_template = [json.loads(f.readlines()[0])['template'],]
        print("Initial random template: ", random_template)
        print("Initial human template: ", human_template)
        savepath = os.path.join(args.output,f'target_baseline/disc-prompt-search_{model_name_parse}_{relation}_{random_seed}_random.tsv') 
        autoprompt.train(random_template, target_dataset, relation, args.n_iterations_max, args.batch_size, savepath)


        exit()
        
        template_tkn = [(autoprompt.template2tokens(random_template), None, 'x'),]
        scores, result = autoprompt.evaluate_candidates(template_tkn, target_dataset, relation, args.batch_size, n_generated_tokens=10, return_pred=True)
        result['str_prompt'] = result.apply(lambda row: autoprompt.model.tokenizer.decode(row['prompt']), axis=1)
        with open(savepath, 'w') as f:
            f.write(result[['label', 'str_prompt', 'pred']].to_string(columns=['label', 'str_prompt', 'pred']))

        # filled_empty = target_dataset.fill_tokenized_template(relation, empty_template, autoprompt.model.tokenizer, set='dev')





        # savepath = os.path.join(args.output,f'disc-prompt-search_{model_name_parse}_{relation}_{random_seed}_random.tsv') 
        # random_template = ['[X]'+model.tokenizer.decode([random.randint(0, model.get_vocab_size()-1) for _ in range(5)])+' [Y]' for __ in range(10)]
        # filled_random = target_dataset.fill_tokenized_template(relation, random_template, autoprompt.model.tokenizer, set='dev')




        # autoprompt.train(initial_template, target_dataset, relation, args.n_iterations_max, args.batch_size, savepath)