import argparse
import torch
from torch.nn.functional import kl_div, cosine_similarity
import pandas as pd
from tqdm import tqdm
import logging
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src.prompt.machine_prompt import DiscreteGradientPromptSearch, OneTokenGradientPromptSearch, RandomPairPromptSearch
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
    parser.add_argument('--paraphrase_path', type=str, default='data/paraphrases/relation-paraphrases_v2.txt')
    parser.add_argument('--lama_path', type=str, default='data/opti-data/autoprompt_data')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--n_iterations_max', type=int, default=100)
    parser.add_argument('--n_population', type=int, default=50)
    parser.add_argument('--num_candidates', type=int, default=5)
    parser.add_argument('--relation', type=str, default='all')
    parser.add_argument('--mode', type=str, default='hot')

    args = parser.parse_args()
    print(args)

    return args

def forward_extract_from_pair(model, templates, input_pair, kn_act_buffer):
    input_pair_tkn = model.tokenizer.encode(input_pair[0])
    filled_templates = [input_pair_tkn+template for template in templates]
    filled_templates = [(torch.tensor(t).unsqueeze(0), torch.ones(len(t)).unsqueeze(0)) for t in filled_templates]
    all_intermediate_representation = []
    all_intermediate_preds = []
    all_intermediate_softmax = []
    all_activations = []
    all_mlp = []
    all_mlp_pred = []
    all_attn = []
    all_attn_pred = []
    for i in range(len(filled_templates)):
        tkn_input = filled_templates[i]
        intermediate_preds, intermediate_softmax, intermediate_representation, knowledge_neurones, mlp, mlp_pred, attn, attn_pred = model.get_intermediate_output(tkn_input, tokenize=False)
        all_intermediate_representation.append(intermediate_representation)
        all_intermediate_preds.append(intermediate_preds)
        all_intermediate_softmax.append(intermediate_softmax)
        all_activations.append(knowledge_neurones)
        all_mlp.append(mlp)
        all_mlp_pred.append(mlp_pred)
        all_attn.append(attn)
        all_attn_pred.append(attn_pred)
    all_intermediate_representation = torch.stack(all_intermediate_representation).float() # shape (prompts, layers, h)
    all_intermediate_softmax = torch.stack(all_intermediate_softmax).float() # shape (prompts, layers, d_voc)
    all_activations = torch.stack(all_activations)
    all_mlp = torch.stack(all_mlp)
    all_attn = torch.stack(all_attn)
    return all_intermediate_preds, all_intermediate_softmax, all_intermediate_representation, all_activations, all_mlp, all_mlp_pred, all_attn, all_attn_pred

# def test_numerical_stability(v):



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

    #load LAMA
    print("Loading LAMA...")
    lamaset = LAMAset(args.lama_path, portion=1.0)

    #load human rephrases
    print("Loading human paraphrases...")
    paraphrases=parse_paraphrases(args.paraphrase_path)        
    # only keep template where '[X]' is the first token / TODO: adress this
    paraphrases={relation:[t for t in templates if t.startswith('[X]')] for relation,templates in paraphrases.items() if len([t for t in templates if t.startswith('[X]')])>0}
    print("Starting!")

    relations = lamaset.get_relations() if args.relation=='all' else [args.relation,]
    relations = ['P1376',]#['P1376', 'P36']

    # initialise the search algo
    # oneTokSearch = OneTokenGradientPromptSearch(model, args.num_candidates, mode=args.mode)
    promptSearch = DiscreteGradientPromptSearch(model, n_population=1, num_candidates=args.num_candidates, n_rounds=2, verbose=True)

    # start iterating across relations
    for relation in relations: # in the future, run all relation in parallel in different scrips
        # initial_template = paraphrases[relation]
        # initial_template = ['[X]'+model.tokenizer.decode([random.randint(0, model.get_vocab_size()-1) for _ in range(8)])+' [Y]',]
        # """
        # dataset is a list of tuple [(X,Y), ...]
        # where X is used to fill in the template and Y is the expected next token.
        # """
        # savepath = os.path.join(args.output,f'one-tok-search-{args.mode}_{model_name_parse}_{relation}_{random_seed}.tsv') 
        # # template_set, df_human, df_candidates = oneTokSearch.search(initial_template, None, lamaset, relation, args.batch_size, only_best_human=True)
        # template_set = promptSearch.train(initial_template, lamaset, relation, 50, args.batch_size, None, only_best_human=True)
        """
        Origin=human
        """
        # template_set = {('[X] , that road the capital city of [Y]', 0.24193548387096775, 'R-2-19-612'), ('[X] LawyersUT anything the capital city of [Y]', 0.3225806451612903, 'R-2-19-626-3253-52348-67101-100741'), ('[X] bell that shield the capital city of [Y]', 0.25806451612903225, 'R-2-20-642'), ('[X]  Vo that is� capital city of [Y]', 0.24193548387096775, 'R-2-50-1797'), ('[X], that is the capital city of [Y]', 0.22580645161290322, 'R-2'), ('[X] bellmilitary propel the capital city of [Y]', 0.3064516129032258, 'R-2-20-642-3478-60391'), ('[X] STA Transactions Strongh the capital city of [Y]', 0.3387096774193548, 'R-2-20-642-3465-59698-67490-115020'), ('[X] LawyersDemon anything the capital city of [Y]', 0.3387096774193548, 'R-2-19-626-3253-52347'), ('[X] bellwing propel the capital city of [Y]', 0.3387096774193548, 'R-2-20-642-3478-60391-67348-109561'), ('[X] ,Keith Hanna the capital city of [Y]', 0.25806451612903225, 'R-2-3-114'), ('[X] , that anything the capital city of [Y]', 0.24193548387096775, 'R-2-19-626'), ('[X] event that is leaving capital city of [Y]', 0.3064516129032258, 'R-2-32-1033-2557-27035'), ('[X] Reference Champ shield the capital city of [Y]', 0.3225806451612903, 'R-2-20-642-2933-40767'), ('[X] Law vaccinatedDemon anything the capital city of [Y]', 0.3387096774193548, 'R-2-19-626-3253-52347-66599-81443'), ('[X] , recreation is31 capital city of [Y]', 0.22580645161290322, 'R-2-41-1435'), ('[X] bell Mile shield the capital city of [Y]', 0.27419354838709675, 'R-2-20-642-3499'), ('[X] , Bri har the capital city of [Y]', 0.24193548387096775, 'R-2-13-473'), ('[X] bellDrag Extensions the capital city of [Y]', 0.3225806451612903, 'R-2-20-642-3485-60684'), ('[X] Tuesday that is anywhere capital city of [Y]', 0.25806451612903225, 'R-2-32-1033'), ('[X] calculKeith Hanna depicted capital city of [Y]', 0.3064516129032258, 'R-2-3-114-2626-29785'), ('[X]  Coverbitcoin Extensions the capital city of [Y]', 0.3064516129032258, 'R-2-20-642-3485-60684-66424-75235'), ('[X] Thursday Directory Festival the capital city of [Y]', 0.3064516129032258, 'R-2-20-649-4102-67562-117283'), ('[X] STA Transactionsundown the capital city of [Y]', 0.3387096774193548, 'R-2-20-642-3465-59698-67490-115018'), ('[X] bellDrag Extensionsaceutical capital city of [Y]', 0.2903225806451613, 'R-2-20-642-3485-60684-67755-125254'), ('[X] SW Cosponsors propel the capital city of [Y]', 0.3387096774193548, 'R-2-20-642-3478-60391-67320-108410'), ('[X] Lawyers celebritieswer the capital city of [Y]', 0.3225806451612903, 'R-2-19-626-3253-52348-67129-101859'), ('[X]  374Senior is31 capital city of [Y]', 0.3064516129032258, 'R-2-41-1436-4170'), ('[X] , chunk is31 capital city of [Y]', 0.22580645161290322, 'R-2-41-1449'), ('[X] Law Theme995 anything the capital city of [Y]', 0.3387096774193548, 'R-2-19-626-3253-52348-67114-101261'), ('[X] Thursday Directory shield the capital city of [Y]', 0.3064516129032258, 'R-2-20-649-4102'), ('[X] LawyersParticip anything the capital city of [Y]', 0.3064516129032258, 'R-2-19-626-3253-52348'), ('[X] Law BASDemon anything the capital city of [Y]', 0.3387096774193548, 'R-2-19-626-3253-52347-66599-81463'), ('[X] Lawyers Transaction anything Serving capital city of [Y]', 0.3548387096774194, 'R-2-19-626-3253-52347-66682-84974'), ('[X] , verse is31 capital city of [Y]', 0.25806451612903225, 'R-2-41-1441'), ('[X]  Hend thatdet the capital city of [Y]', 0.22580645161290322, 'R-2-12-399'), ('[X] ,Senior 16731 capital city of [Y]', 0.27419354838709675, 'R-2-41-1436-3089-46563'), ('[X] Tuesdayaddr is anywhere Junction city of [Y]', 0.3387096774193548, 'R-2-32-1033-2596-28403-68771'), ('[X] Law affirmationDemon anything the capital city of [Y]', 0.3387096774193548, 'R-2-19-626-3253-52347-66615-82082'), ('[X] Heaven flowers that shield the capital city of [Y]', 0.3064516129032258, 'R-2-20-642-2947-41235'), ('[X] ,Senior is31 capital city of [Y]', 0.27419354838709675, 'R-2-41-1436'), ('[X] LawyersIS anything the capital city of [Y]', 0.3387096774193548, 'R-2-19-626-3253-52348-67101-100742'), ('[X] STA Transactions shield the capital city of [Y]', 0.3225806451612903, 'R-2-20-642-3465-59698'), ('[X] , that is� capital city of [Y]', 0.24193548387096775, 'R-2-50'), ('[X] rec TobiasSenior is31 capital city of [Y]', 0.3225806451612903, 'R-2-41-1436-3032-44291'), ('[X] Tuesday that is oversees capital city of [Y]', 0.3225806451612903, 'R-2-32-1033-2571-27620'), ('[X] Thursday that shield the capital city of [Y]', 0.25806451612903225, 'R-2-20-649'), ('[X] dpodcastal TobiasSenior is31 capital city of [Y]', 0.2903225806451613, 'R-2-41-1436-3032-44291-66311-70658'), ('[X] ThursdayStorage shield the capital city of [Y]', 0.3387096774193548, 'R-2-20-649-4102-67535'), ('[X] LawyersDemon anything buttons capital city of [Y]', 0.3225806451612903, 'R-2-19-626-3253-52347-68101'), ('[X] ins mobsSenior is31 capital city of [Y]', 0.2903225806451613, 'R-2-41-1436-3046-44909'), ('[X] theatImpilitary propel the capital city of [Y]', 0.3387096774193548, 'R-2-20-642-3478-60391-67341-109351'), ('[X] Tuesdayaddr is anywhere Junction city of [Y]', 0.3387096774193548, 'R-2-32-1033-2596-28403'), ('[X] recSpellSenior is31 capital city of [Y]', 0.3064516129032258, 'R-2-41-1436-3032-44283')}
        """
        Origin=random
        """
        # template_set = {('[X] Football Recommraining Tags bearings – True� history ├── [Y]', 0.24193548387096775, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4223-4635'), ('[X], that was formulated in [Y]', 666.0, 'U'), ('[X] Football Recommraining Tags unfortunately – True visionsides ├── [Y]', 0.22580645161290322, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-4076'), ('[X] q Autraining Lu unfortunately – Typical visions Crossref ├── [Y]', 0.11290322580645161, 'R-0-27-162-529-1009-1246-1989-2158-2734-2950'), ('[X] Ink salvage rapists bluff effectiveness108 directly hazard Yin fused [Y]', 666.0, 'R'), ('[X] qutraining Lu unfortunately ", Truelive Crossref ├── [Y]', 0.04838709677419355, 'R-0-27-162-529-1009-1246-1825'), ('[X] counts as a legal term in [Y]', 666.0, 'U'), ('[X] is follower of [Y]', 666.0, 'U'), ('[X] has a citizenship of [Y]', 666.0, 'U'), ('[X] Football happenedraining Tags unfortunately – Trueanting history ├── [Y]', 0.24193548387096775, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4186-4602'), ('[X] Football Recommraining Tags trap – True� history ├── [Y]', 0.24193548387096775, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4223-4752'), ('[X] qutraining Lu unfortunately ", True Aadhaar Crossref reconnect [Y]', 0.0, 'R-0-27-162-529-1009'), ('[X]  GCCoted Lucutting ", Truealogy 105 reconnect [Y]', 0.0, 'R-0-2-208'), ('[X] quoted testament Ghost ", Truealogy juven reconnect [Y]', 0.0, 'R-0-31-345'), ('[X] Tankutraining Lu unfortunately – True visions Crossref ├── [Y]', 0.14516129032258066, 'R-0-27-162-529-1009-1246-1989-2158-2795'), ('[X] Football Recommraining Template unfortunately – True visions Crossref ├── [Y]', 0.20967741935483872, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3368'), ('[X] can be found in [Y]', 666.0, 'U'), ('[X] qutraining Lu unfortunately TRA True Aadhaar Crossref ├── [Y]', 0.04838709677419355, 'R-0-27-162-529-1009-1246-1789'), ("[X]'s headquarters are in [Y]", 666.0, 'U'), ('[X] qutraining binge unfortunately – True visions Crossref ├── [Y]', 0.12903225806451613, 'R-0-27-162-529-1009-1246-1989-2158-2559'), ('[X] Football happenedraining Tags unfortunately – True overriding593 ├── [Y]', 0.24193548387096775, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4186-4605-5050'), ('[X] quoted Lu Ghost glob Truealogyogan reconnect [Y]', 0.0, 'R-0-31-57-729-1181'), ('[X] Football happened bars Tags unfortunately – TrueResource history ├── [Y]', 0.27419354838709675, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4186-4608-4962'), ('[X], located in [Y]', 666.0, 'U'), ('[X] plays [Y]', 666.0, 'U'), ('[X] Football Recommraining Lu unfortunately – True visions Crossref ├── [Y]', 0.14516129032258066, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226'), ('[X] Football happenedraining Tags unfortunately – True overriding history ├── [Y]', 0.24193548387096775, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4186-4605'), ('[X] qutraining Lu unfortunately ", True Aadhaar Crossref ├── [Y]', 0.016129032258064516, 'R-0-27-162-529-1009-1246'), ('[X] qutraining Lu unfortunately – Typical visions Crossref ├── [Y]', 0.12903225806451613, 'R-0-27-162-529-1009-1246-1989-2158-2734'), ('[X] qutraining Lu unfortunately ", True lake Crossref ├── [Y]', 0.08064516129032258, 'R-0-27-162-529-1009-1246-1872'), ('[X], who works as [Y]', 666.0, 'U'), ('[X] qutraining Lu unfortunately ", Truealogy Crossref reconnect [Y]', 0.0, 'R-0-27-162-529'), ('[X] Football Marriottraining Tags unfortunately – True visions history ├── [Y]', 0.22580645161290322, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4182'), ('[X] qutraining Lu unfortunately ", Trueublished Crossref ├── [Y]', 0.08064516129032258, 'R-0-27-162-529-1009-1246-1912'), ('[X] recorded for [Y]', 666.0, 'U'), ('[X], that develops [Y]', 666.0, 'U'), ('[X] qutraining Lu unfortunately ", Truecampaign Crossref ├── [Y]', 0.06451612903225806, 'R-0-27-162-529-1009-1246-1917'), ('[X] Football Recommraining Tags unfortunately – True� history ├── [Y]', 0.24193548387096775, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4223'), ('[X] Football Recommraining Tags unfortunately – True visions Crossref ├── [Y]', 0.20967741935483872, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362'), ('[X] quoted Lu Ghost ",heddaralogy 105 reconnect [Y]', 0.0, 'R-0-17'), ('[X] Football Recommraining Template unfortunately – True visionsisions ├── [Y]', 0.1935483870967742, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3368-3754'), ('[X] Football Recommraining Tags unfortunately – True Chain history ├── [Y]', 0.22580645161290322, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4223-4648'), ('[X] is written in [Y]', 666.0, 'U'), ('[X] quoted Lu Ghost ",434alogy 105 reconnect [Y]', 0.0, 'R-0-20'), ('[X] qutraining elementary unfortunately – True visions Crossref ├── [Y]', 0.12903225806451613, 'R-0-27-162-529-1009-1246-1989-2158-2552'), ('[X] java Monstrous revenue Success publishedilst Soldiersologically occassch [Y]', 666.0, 'R'), ('[X] qutraining Lu Lionel ", Trueongo Crossref reconnect [Y]', 0.0, 'R-0-27-162-549-867'), ('[X] was developed by [Y]', 666.0, 'U'), ('[X] qutraininguesday Ghost ", Trueaughters Crossref reconnect [Y]', 0.0, 'R-0-27-162-544-971'), ('[X] qutraining Lu unfortunately ", True705 Crossref ├── [Y]', 0.04838709677419355, 'R-0-27-162-529-1009-1246-1919'), ('[X] Football Recommraining Tags unfortunately – True visions primer ├── [Y]', 0.22580645161290322, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3703-4391'), ('[X] Football Recommraining Tags unfortunately – Trueimposed Crossref ├── [Y]', 0.22580645161290322, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3894'), ('[X] quoted Planes Ghost glob Truealogy juven reconnect [Y]', 0.0, 'R-0-31-57-729-1158'), ('[X] qutraining Lu unfortunately – True visions Crossref ├── [Y]', 0.12903225806451613, 'R-0-27-162-529-1009-1246-1989-2158'), ('[X] Football happenedraining Tagshetically – TrueResource history ├── [Y]', 0.22580645161290322, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4186-4608-4991'), ('[X] Football Recommraining Tags unfortunately – True commenters primer ├── [Y]', 0.22580645161290322, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3703-4391-4809'), ('[X] is a member of [Y]', 666.0, 'U'), ('[X] is to debut on [Y]', 666.0, 'U'), ('[X] validated Chemicalainer sidewaysedi supplemental Corinth traveller Render prec [Y]', 666.0, 'R'), ('[X] was started in [Y]', 666.0, 'U'), ('[X] was called after [Y]', 666.0, 'U'), ('[X] quotedGar Ghost ",434alogy 105 reconnect [Y]', 0.0, 'R-0-20-237'), ('[X] quoted Lu Ghost ", behavioralalogy Garcia reconnect [Y]', 0.0, 'R-0-25-315'), ('[X] qutraining Lu Learned – True visions Crossref ├── [Y]', 0.12903225806451613, 'R-0-27-162-529-1009-1246-1989-2158-2563'), ('[X] qu supra Lu Ghost 246heddarcsv 105 reconnect [Y]', 0.0, 'R-0-17-134-396-805'), ('[X] owner [Y]', 666.0, 'U'), ('[X] qu saturation Lu Ghost ", Truealogy Garcia reconnect [Y]', 0.0, 'R-0-25-310'), ('[X] q Recommraining Lu unfortunately – True visions Crossref ├── [Y]', 0.12903225806451613, 'R-0-27-162-529-1009-1246-1989-2158-2588'), ('[X], which has the capital city [Y]', 666.0, 'U'), ('[X] qutraining Lu Ghost ", Trueaughters Crossref0000000000000000 [Y]', 0.0, 'R-0-27-162-544-987'), ('[X] found employment in [Y]', 666.0, 'U'), ('[X] Tankutraining Lu unfortunately – restrictive visions Crossref ├── [Y]', 0.12903225806451613, 'R-0-27-162-529-1009-1246-1989-2158-2795-2874'), ('[X] Football happened bars Tags unfortunately – True Simpson history ├── [Y]', 0.25806451612903225, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4186-4608-4962-5439'), ('[X] Football Recommraining Tags unfortunately – True shocks relating ├── [Y]', 0.20967741935483872, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-4122-4166'), ('[X] Football Recommraining Tags unfortunately – True visions relating ├── [Y]', 0.24193548387096775, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-4122'), ('[X] Football Recommraining Tags unfortunately – Trueymaps ├── [Y]', 0.22580645161290322, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3910-4307'), ('[X] plays the following music genre [Y]', 666.0, 'U'), ('[X], that is the capital of [Y]', 666.0, 'U'), ('[X] and its sister city [Y]', 666.0, 'U'), ('[X] Football Recommraining Tags unfortunately – True visions history ├── [Y]', 0.24193548387096775, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706'), ('[X] is employed by [Y]', 666.0, 'U'), ('[X] is part of [Y]', 666.0, 'U'), ('[X] Football Recommraining Tags unfortunately – True visionside ├── [Y]', 0.22580645161290322, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3707'), ('[X] qut Historically Lu Learned – True visions Crossref ├── [Y]', 0.14516129032258066, 'R-0-27-162-529-1009-1246-1989-2158-2563-3131'), ('[X], who had the official role of [Y]', 666.0, 'U'), ('[X] was originally performed in [Y]', 666.0, 'U'), ('[X] was originally from [Y]', 666.0, 'U'), ('[X] quoted Lu Ghost N Truealogy juven reconnect [Y]', 0.0, 'R-0-31-57'), ('[X] Football happened schedule Tags unfortunately – TrueResource history ├── [Y]', 0.24193548387096775, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4186-4608-4962-5393'), ('[X] quoted Lucutting ", Truealogy 105 reconnect [Y]', 0.0, 'R-0-2'), ('[X]  Brothers saturation Lu Ghost ", Truealogy Garcia reconnect [Y]', 0.0, 'R-0-25-310-448'), ('[X] plays as [Y]', 666.0, 'U'), ('[X] quoted Lu Ghost 246heddarcsv 105 reconnect [Y]', 0.0, 'R-0-17-134-396'), ('[X] quoted Lu Ghost ", bashingalogy 105 reconnect [Y]', 0.0, 'R-0-16'), ('[X] Football happened bars Tags vacancy – TrueResource history ├── [Y]', 0.25806451612903225, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4186-4608-4962-5370'), ('[X] thinkdirty obsessed TRA Greece Types Krildodescription Unemployment [Y]', 666.0, 'R'), ('[X] quoted Lu Ghost ",heddarcsv 105 reconnect [Y]', 0.0, 'R-0-17-134'), ('[X] Football happenedraining Tags heritage – Trueanting history ├── [Y]', 0.22580645161290322, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4186-4602-5102'), ('[X]ifleintentionawei Payton Lisbon Station ClusterAdv 1991adder [Y]', 666.0, 'R'), ('[X] shares border with [Y]', 666.0, 'U'), ('[X] Survivors salary Bert Tasassembled wiser automate reportedly Gas stride [Y]', 666.0, 'R'), ('[X] Football happenedraining Tags unfortunately – TrueResource history ├── [Y]', 0.25806451612903225, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4186-4608'), ('[X] quoted Lu Ghost ", Truealogy 105 reconnect [Y]', 0.0, 'R-0'), ('[X] quoted Lu Ghost ", Truealogy Colossus reconnect [Y]', 0.0, 'R-0-29'), ('[X] Football happenedraining Tags unfortunately – True visions history ├── [Y]', 0.24193548387096775, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4186'), ('[X] quoted Lu Ghost ", coachingalogy 105 reconnect [Y]', 0.0, 'R-0-19'), ('[X] qutraining Lu unfamiliar – True visions Crossref ├── [Y]', 0.12903225806451613, 'R-0-27-162-529-1009-1246-1989-2158-2563-3141'), ('[X] Sending HTC quart Exceptmust!] Franc Monica Eva haven [Y]', 666.0, 'R'), ('[X] Football Recommraining Tags unfortunately – True visionsimer ├── [Y]', 0.22580645161290322, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3703'), ('[X] UberCCoted Lucutting ", True je 105 reconnect [Y]', 0.0, 'R-0-2-208-644-798'), ('[X] UberCCoted Lucutting ", Truealogy 105 reconnect [Y]', 0.0, 'R-0-2-208-644'), ('[X] Isaiah Volcano cyclon Replricted teachers moisturssh caliber [Y]', 666.0, 'R'), ('[X] Football Recommraining Tags trap – True commodities history ├── [Y]', 0.22580645161290322, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4223-4752-5243'), ('[X] used to communicate in [Y]', 666.0, 'U'), ('[X] qutraining Fif unfortunately – True visions Crossref ├── [Y]', 0.12903225806451613, 'R-0-27-162-529-1009-1246-1989-2158-2557'), ('[X] qutraining Lu Ghost ", Trueongo Crossref reconnect [Y]', 0.0, 'R-0-27-162-549'), ('[X] qutThings Lu unfortunately – Typical visions Crossref ├── [Y]', 0.12903225806451613, 'R-0-27-162-529-1009-1246-1989-2158-2734-2958'), ('[X] died in [Y]', 666.0, 'U'), ('[X] Football happened bars Tags unfortunately –fanResource history ├── [Y]', 0.25806451612903225, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4186-4608-4962-5427'), ('[X] Hundred Swan Derek Tosh nurtrio Bos mafiaSu~~ [Y]', 666.0, 'R'), ('[X] qissionraining Lu unfortunately – True visions Crossref ├── [Y]', 0.12903225806451613, 'R-0-27-162-529-1009-1246-1989-2158-2755'), ('[X] Ors saturation Lu Ghost ", Truealogy Garcia reconnect [Y]', 0.0, 'R-0-25-310-448-1103'), ('[X] quoted Lu Ghost ", Truealogy juven reconnect [Y]', 0.0, 'R-0-31'), ('[X] quoted Lu Ghost glob Truealogy juven reconnect [Y]', 0.0, 'R-0-31-57-729'), ('[X] q Sprraining Lu Learned – True visions Crossref ├── [Y]', 0.14516129032258066, 'R-0-27-162-529-1009-1246-1989-2158-2563-3120'), ('[X] Football Recommraining Tags unfortunately – True visionsaps ├── [Y]', 0.22580645161290322, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3910'), ('[X] Football Recommraining Tags unfortunately – True daring history ├── [Y]', 0.22580645161290322, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4227'), ('[X] Football happenedraining Tags unfortunately – True Parables history ├── [Y]', 0.22580645161290322, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4186-4608-5015'), ('[X] Football occupraining Tags trap – True� history ├── [Y]', 0.25806451612903225, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4223-4752-5199'), ('[X], a subclass of [Y]', 666.0, 'U'), ('[X] Football happenedraining Tags unfortunately – Trueanship history ├── [Y]', 0.24193548387096775, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4186-4602-5150'), ('[X] quoted Lu Ghost ", graphenealogy juven reconnect [Y]', 0.0, 'R-0-31-69'), ('[X] qutraining Lu Ghost ", Trueaughters Crossref reconnect [Y]', 0.0, 'R-0-27-162-544'), ('[X] qutraining Fif unfortunately – Truechery Crossref ├── [Y]', 0.12903225806451613, 'R-0-27-162-529-1009-1246-1989-2158-2557-3104'), ('[X] ogenousutraining binge unfortunately – True visions Crossref ├── [Y]', 0.11290322580645161, 'R-0-27-162-529-1009-1246-1989-2158-2559-2999'), ('[X] quoted testament GhostKevin Truealogy juven reconnect [Y]', 0.0, 'R-0-31-345-569'), ('[X] has diplomatic ties with [Y]', 666.0, 'U'), ('[X] quoted testament GhostKevin Truealogypotion reconnect [Y]', 0.0, 'R-0-31-345-569-1085'), ('[X] qutraining Lu Ghost ", Truealogy Crossref reconnect [Y]', 0.0, 'R-0-27-162'), ('[X] works in the field of [Y]', 666.0, 'U'), ('[X] deemed Ack Hawhe taxingwheel entiretyusc dessert Lloyd [Y]', 666.0, 'R'), ('[X] quotedGar Ghost ", Hugalogy 105 reconnect [Y]', 0.0, 'R-0-20-237-619'), ('[X] qutraining Lu unfortunately ", True visions Crossref ├── [Y]', 0.06451612903225806, 'R-0-27-162-529-1009-1246-1989'), ('[X] Football Recommraining Tags unfortunately defe True� history ├── [Y]', 0.22580645161290322, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4223-4690'), ('[X] Football Recommraining Tags unfortunately – True Older history ├── [Y]', 0.24193548387096775, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4223-4651'), ('[X] quoted Lu Ghost fictheddaralogy 105 reconnect [Y]', 0.0, 'R-0-17-112'), ('[X] quoted Lu Ghost ", Truealogy Crossref reconnect [Y]', 0.0, 'R-0-27'), ('[X] is in [Y]', 666.0, 'U'), ('[X] existingutraining Lu unfortunately – True visions Crossref ├── [Y]', 0.12903225806451613, 'R-0-27-162-529-1009-1246-1989-2158-2794'), ('[X] Football happened bars Tags unfortunately – True helpful history ├── [Y]', 0.25806451612903225, 'R-0-27-162-529-1009-1246-1989-2158-2588-3226-3362-3706-4186-4608-4962-5440'), ('[X] is a part of the continent of [Y]', 666.0, 'U'), ('[X] quoted Lu Ghost ", Truealogy Garcia reconnect [Y]', 0.0, 'R-0-25')}
        # in the following, mutation are really successusives 
        # template_set = {('[X] unofficial debatesstroke ferocious deadlines436380 copyright [Y]', 0.064516129032258, 'R-0-25-518-639-1450'), ('[X] unofficial debatesstroke ferocious deadlines Breast380 copyright [Y]', 0.032258064516129, 'R-0-25-518-639'), ('[X] populations widening vulgarphalt bible Garner detailingHomeUrl Photography Wonderful [Y]', 0.2741935483870967, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770'), ('[X] passers exhibition debates identity detailing deadlinesUrl ideas copyright [Y]', 0.1612903225806451, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510'), ('[X] populations widening vot marryingokes Garner Greater1900Url Photography Wonderful [Y]', 0.2419354838709677, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449-25428-26390-26483-28063-28420-28620'), ('[X] unofficial debates Data ferocious deadlines Breast380 copyright [Y]', 0.032258064516129, 'R-0-25-518'), ('[X] populations stockpsteroubtedly } Garner Greater1900Url Photography Wonderful [Y]', 0.3064516129032258, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449-25428-26390-26483-28063-28420-28620-29737-30354-31143-32014'), ('[X] populations stockpuerAYさwealth PepislUrl Photography Wonderful [Y]', 0.2741935483870967, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449-25428-26390-26483-28063-28420-28620-29737-30354-31143-32014-33224-33621-34649-34982-35484-36413-36588-37852-38558-42910-43010-43833-44492-45835-46480'), ('[X] populations widening magn marryingokes Garner agesHomeUrl Photography Wonderful [Y]', 0.2903225806451613, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449-25428-26390-26483'), ('[X] refillers exhibition debates identity detailing reservationsUrl machine Wonderful [Y]', 0.2580645161290322, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051'), ('[X] populations stockpuerAYさ inclusion Greater019Url Photography Wonderful [Y]', 0.2741935483870967, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449-25428-26390-26483-28063-28420-28620-29737-30354-31143-32014-33224-33621-34649-34982-35484-36413-36588-37852-38558-42910-43010'), ('[X] populations stockpoken comprised } Garner Greater1900Url Photography Wonderful [Y]', 0.3064516129032258, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449-25428-26390-26483-28063-28420-28620-29737-30354-31143-32014-33224-33621'), ('[X] populations widening vot marryingokes Garner ages1900Url Photography Wonderful [Y]', 0.2903225806451613, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449-25428-26390-26483-28063-28420'), ('[X] populations Brooksersphalt debates maximum detailingHomeUrl Photography Wonderful [Y]', 0.2419354838709677, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039'), ('[X] 246 frontrunner debates dipping Alma deadlines436380 copyright [Y]', 0.0806451612903225, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035'), ('[X] populations stockpuerAY } Garner Greater019Url Photography Wonderful [Y]', 0.2903225806451613, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449-25428-26390-26483-28063-28420-28620-29737-30354-31143-32014-33224-33621-34649-34982-35484-36413-36588'), ('[X] populations widening magnmetokes Garner agesHomeUrl Photography Wonderful [Y]', 0.2741935483870967, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449-25428'), ('[X] turns VIS debates dipping detailing deadlines436380 copyright [Y]', 0.0806451612903225, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172'), ('[X] populationsillersphalt debates maximum detailing complaintsUrl Photography Wonderful [Y]', 0.2580645161290322, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176'), ('[X] unofficial debates dipping Alma deadlines436380 copyright [Y]', 0.0483870967741935, 'R-0-25-518-639-1450-2007-2222-3016-3728'), ('[X] turns VIS debates identity detailing deadlines436380 copyright [Y]', 0.0806451612903225, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781'), ('[X] turns VIS debates identity detailing deadlinesUrl ideas copyright [Y]', 0.1290322580645161, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216'), ('[X] populations lifestylesacedAYさwealth 225islUrl Photography Wonderful [Y]', 0.2741935483870967, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449-25428-26390-26483-28063-28420-28620-29737-30354-31143-32014-33224-33621-34649-34982-35484-36413-36588-37852-38558-42910-43010-43833-44492-45835-46480-46615-48058-48498-48814'), ('[X] populations stockpuer herein } Garner Greater1900Url Photography Wonderful [Y]', 0.2903225806451613, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449-25428-26390-26483-28063-28420-28620-29737-30354-31143-32014-33224-33621-34649-34982-35484'), ('[X] 246 frontrunner debates dipping detailing deadlines436380 copyright [Y]', 0.0967741935483871, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621'), ('[X] unofficial bureaucratic Data ferocious deadlines Breast380 Darwin [Y]', 0.0, 'R-0'), ('[X] populations stockpuer comprised } Garner Greater1900Url Photography Wonderful [Y]', 0.3064516129032258, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449-25428-26390-26483-28063-28420-28620-29737-30354-31143-32014-33224-33621-34649-34982'), ('[X] passers assembled debates identity detailing deadlinesUrl ideas copyright [Y]', 0.1129032258064516, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425'), ('[X] populations wideningsteroubtedlyokes Garner Greater1900Url Photography Wonderful [Y]', 0.3064516129032258, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449-25428-26390-26483-28063-28420-28620-29737-30354'), ('[X] refillers exhibition debates nerves detailing complaintsUrl machine Wonderful [Y]', 0.2580645161290322, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061'), ('[X] refillers exhibition debates identity detailing deadlinesUrl ideas Wonderful [Y]', 0.1935483870967742, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604'), ('[X] refillers exhibition debates identity detailing desiresUrl machine Wonderful [Y]', 0.2419354838709677, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177'), ('[X] unofficial debates dis Alma deadlines436380 copyright [Y]', 0.032258064516129, 'R-0-25-518-639-1450-2007-2222-3016'), ('[X] populations stockpuerAY } inclusion Greater019Url Photography Wonderful [Y]', 0.3064516129032258, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449-25428-26390-26483-28063-28420-28620-29737-30354-31143-32014-33224-33621-34649-34982-35484-36413-36588-37852-38558'), ('[X] populations widening magnphaltokes Garner agesHomeUrl Photography Wonderful [Y]', 0.2741935483870967, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449'), ('[X] populationsillers exhibition debates maximum detailing complaintsUrl Photography Wonderful [Y]', 0.2741935483870967, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077'), ('[X] populations Brooks Hughesphalt debated maximum detailingHomeUrl Photography Wonderful [Y]', 0.2741935483870967, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552'), ('[X] populations widening Hughesphalt bible maximum detailingHomeUrl Photography Wonderful [Y]', 0.2580645161290322, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357'), ('[X] populations collapsing Hughesphalt debated maximum detailingHomeUrl Photography Wonderful [Y]', 0.2580645161290322, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227'), ('[X] populations stockpuerAYさwealth Pep019Url Photography Wonderful [Y]', 0.2903225806451613, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449-25428-26390-26483-28063-28420-28620-29737-30354-31143-32014-33224-33621-34649-34982-35484-36413-36588-37852-38558-42910-43010-43833-44492'), ('[X] populations stockpuerAYさwealth 225islUrl Photography Wonderful [Y]', 0.2903225806451613, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242-23449-25428-26390-26483-28063-28420-28620-29737-30354-31143-32014-33224-33621-34649-34982-35484-36413-36588-37852-38558-42910-43010-43833-44492-45835-46480-46615-48058'), ('[X] refillers exhibition debates maximum detailing complaintsUrl arrange Wonderful [Y]', 0.2580645161290322, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868'), ('[X] populations widening magnphalt bible Garner agesHomeUrl Photography Wonderful [Y]', 0.2741935483870967, 'R-0-25-518-639-1450-2007-2222-3016-3728-4670-5035-5621-5961-6172-7124-7781-8515-9216-9297-9425-10106-10510-11381-11604-12191-13177-13228-14051-14552-15061-15279-15868-15918-16077-17176-17703-18039-18483-18552-19227-19936-20357-20826-21770-22854-23242'), ('[X] unofficial debates dis ferocious deadlines436380 copyright [Y]', 0.0967741935483871, 'R-0-25-518-639-1450-2007-2222')}

        template_set = [([13075, 19850, 14545, 39979, 22671, 6471, 33596, 47805], 0.0, 'R-0'), ([13075, 38200, 14545, 39979, 29756, 6471, 33596, 47805], 0.0, 'R-0-3-7'), ([43681, 38200, 37471, 39979, 29756, 6471, 33596, 47805], 0.0, 'R-0-3-7-12-7'), ([43681, 37331, 37471, 3834, 29756, 6471, 33596, 47805], 0.0, 'R-0-3-7-12-7-2-7'), ([43681, 37331, 37471, 3834, 29756, 26272, 42935, 47805], 0.0, 'R-0-3-7-12-7-2-7-29-29'), ([9743, 37331, 37471, 3834, 29756, 26272, 42935, 47805], 0.0, 'R-0-3-7-12-7-2-7-29-29-4-10'), ([43560, 37331, 37471, 3834, 29756, 6740, 42935, 47805], 0.0, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2'), ([43560, 12910, 37471, 3834, 29756, 6740, 12493, 47805], 0.0, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25'), ([43560, 12910, 37471, 19275, 29756, 6740, 31743, 47805], 0.0, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27'), ([43560, 12910, 37471, 19275, 29756, 6740, 44899, 30479], 0.0, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35'), ([34923, 12910, 37471, 19275, 29756, 6740, 48545, 30479], 0.03225806451612903, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3'), ([34923, 12910, 6135, 19275, 29756, 6740, 49955, 30479], 0.03225806451612903, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2'), ([34923, 32317, 6135, 19275, 29756, 6740, 49955, 30479], 0.03225806451612903, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8'), ([34923, 13287, 6135, 19275, 29756, 6740, 6592, 30479], 0.03225806451612903, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8'), ([34923, 13287, 6135, 19275, 29756, 6740, 49762, 22402], 0.11290322580645161, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20'), ([153, 13287, 6135, 19275, 22178, 6740, 49762, 22402], 0.14516129032258066, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2'), ([153, 11444, 891, 19275, 22178, 6740, 49762, 22402], 0.1774193548387097, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8'), ([153, 2824, 891, 19275, 22178, 6740, 49762, 22402], 0.1774193548387097, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5'), ([153, 2824, 891, 19275, 22178, 6740, 19080, 22402], 0.1935483870967742, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24'), ([153, 2824, 891, 35477, 22178, 6740, 19080, 22402], 0.1935483870967742, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31'), ([153, 31444, 20437, 35477, 22178, 6740, 19080, 22402], 0.14516129032258066, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14'), ([153, 31444, 20437, 198, 22178, 6159, 19080, 22402], 0.1774193548387097, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15'), ([153, 31444, 20437, 198, 22178, 6159, 49556, 22402], 0.22580645161290322, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26'), ([12195, 18208, 20437, 198, 22178, 6159, 49556, 22402], 0.24193548387096775, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4'), ([1596, 18208, 9951, 198, 22178, 6159, 49556, 22402], 0.24193548387096775, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9'), ([39685, 18208, 9951, 198, 22178, 6159, 49556, 22402], 0.22580645161290322, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11'), ([37214, 18208, 9951, 23922, 22178, 6159, 49556, 22402], 0.24193548387096775, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11'), ([4245, 18208, 24629, 23922, 22178, 6159, 49556, 22402], 0.3064516129032258, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8'), ([4245, 28662, 24629, 23076, 22178, 6159, 49556, 22402], 0.2903225806451613, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10'), ([4245, 28662, 24629, 23076, 22178, 6159, 49556, 22402], 0.2903225806451613, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10'), ([4245, 28662, 24629, 2807, 22178, 6159, 49556, 22402], 0.2903225806451613, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30'), ([4245, 9444, 24629, 2807, 22178, 6159, 49556, 22402], 0.3064516129032258, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3'), ([4245, 20561, 24629, 2807, 22178, 6159, 49556, 22402], 0.3064516129032258, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4'), ([4245, 20561, 24629, 2807, 22178, 6159, 49556, 22402], 0.3064516129032258, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4'), ([4245, 37959, 24629, 2807, 22178, 6159, 49556, 22402], 0.2903225806451613, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6'), ([4245, 28251, 24629, 2807, 22178, 6159, 49556, 22402], 0.3064516129032258, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6-14-3'), ([4245, 28251, 24629, 2807, 22178, 6159, 49556, 22402], 0.3064516129032258, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6-14-3'), ([4245, 28251, 24629, 2807, 22245, 6159, 49556, 22402], 0.3064516129032258, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6-14-3-8-25'), ([4245, 28251, 24629, 2807, 24741, 6159, 49556, 22402], 0.3225806451612903, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6-14-3-8-25-4-14'), ([4245, 28251, 24629, 22830, 30012, 6159, 49556, 22402], 0.3064516129032258, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6-14-3-8-25-4-14-4-19'), ([4245, 28251, 24629, 22830, 30012, 6159, 49556, 22402], 0.3064516129032258, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6-14-3-8-25-4-14-4-19'), ([4245, 28251, 24629, 22830, 30012, 6159, 49556, 22402], 0.3064516129032258, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6-14-3-8-25-4-14-4-19'), ([4245, 28251, 24629, 22830, 48174, 6159, 49556, 22402], 0.3064516129032258, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6-14-3-8-25-4-14-4-19-3'), ([8166, 28251, 24629, 22830, 48174, 6159, 49556, 22402], 0.3064516129032258, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6-14-3-8-25-4-14-4-19-3-4'), ([18817, 28251, 24629, 22830, 48174, 6159, 49556, 22402], 0.3064516129032258, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6-14-3-8-25-4-14-4-19-3-4-6-4'), ([46843, 28251, 24629, 22830, 48174, 6159, 49556, 22402], 0.3387096774193548, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6-14-3-8-25-4-14-4-19-3-4-6-4-10-5'), ([46843, 28251, 24629, 22830, 48174, 6159, 49556, 22402], 0.3387096774193548, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6-14-3-8-25-4-14-4-19-3-4-6-4-10-5'), ([46843, 28251, 24629, 22830, 48174, 6159, 49556, 22402], 0.3387096774193548, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6-14-3-8-25-4-14-4-19-3-4-6-4-10-5'), ([46843, 28251, 24629, 22830, 48174, 6159, 49556, 22402], 0.3387096774193548, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6-14-3-8-25-4-14-4-19-3-4-6-4-10-5'), ([28989, 28251, 24629, 26608, 48174, 6159, 49556, 22402], 0.27419354838709675, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6-14-3-8-25-4-14-4-19-3-4-6-4-10-5-19-3'), ([28989, 28251, 24629, 25756, 48174, 6159, 49556, 22402], 0.3225806451612903, 'R-0-3-7-12-7-2-7-29-29-4-10-32-2-19-25-12-27-29-35-29-3-21-2-3-8-28-8-24-20-26-2-19-8-3-5-19-24-16-31-10-14-6-15-26-5-4-9-9-7-11-16-11-2-8-2-10-30-19-3-9-4-6-14-3-8-25-4-14-4-19-3-4-6-4-10-5-19-3-17')]

        # print(template_set)

        # promptSearch.save(template_set, 1, os.path.join(args.output, 'template_search.tsv'))

        # add random templates
        avg_prompt_len = sum([len(t[0]) for t in template_set])/len(template_set)
        print("Average len: ", avg_prompt_len)

        # adding random prompts
        random_prompts = [([random.randint(0, model.get_vocab_size()-1) for _ in range(int(avg_prompt_len))], 666.0, 'R') for z in range(10)]
        template_set += random_prompts

        # adding unrelated prompts
        unrelated_prompts = [promptSearch.template2tokens(random.sample(paraphrases[r], 1)[0]) for r in paraphrases]
        unrelated_prompts = [(t, 0.0, 'U') for t in unrelated_prompts]
        template_set += unrelated_prompts

        # adding human (related) templates
        human_templates = [(promptSearch.template2tokens(t), 666.0, 'H') for t in paraphrases[relation]]
        template_set += human_templates

        # adding machine random
        # randomPromptSearch = RandomPairPromptSearch(model, 50, 10, n_rounds=2, verbose=True)
        # todo later

        # print(f"[{relation}] Templates:", template_set)
        df_templates = pd.DataFrame(template_set, columns=['template', 'accuracy', 'tid'])# ,'h_agreement'])
        df_templates['str_template'] = df_templates.apply(lambda row: promptSearch.tokens2template(row['template']), axis=1)

        # keep topk most accurates
        # df_templates = df_templates.nlargest(5, 'accuracy')
        # remove template with low accuracy
        # df_templates = df_templates[df_templates['accuracy'] > 0.1]
        # compute PPL for each template
        tid2n = {'U':-1, 'R':-2, 'H':-3}
        df_templates['ppl'] = model.compute_ppl_from_tokens_batch(
            df_templates['template'].tolist(), None
            ).tolist()
        # df_templates['ppl'] = df_templates.apply(lambda row: model.compute_ppl_from_tokens_batch(torch.tensor(row['template']), torch.ones(len(row['template']))), axis=1)
        df_templates['n_mutations'] = df_templates.apply(lambda row: (len(row['tid'].split('-'))-2) if '-' in row['tid'] else tid2n[row['tid']], axis=1)
        print(df_templates)
        # print
        print(df_templates.sort_values(by='ppl').to_string(index=False))
        
        # plot
        g = sns.scatterplot(data=df_templates, y='accuracy', x='ppl')
        g.set_xscale('log')
        g.figure.savefig(f'acc_ppl_{model_name_parse}_{relation}.png')
        plt.clf()

        # lama
        lama_samples = lamaset.sample_relation(relation, 'dev', 10)

        # filter machine to only keep to last ones
        df_templates['type'] = df_templates.apply(lambda row: 'M' if 'R-' in row['tid'] else row['tid'], axis=1)
        cond_last_machine = df_templates['n_mutations']>df_templates['n_mutations'].quantile(0.90)
        cond_not_machine = df_templates['type']!='M'
        df_templates = df_templates[cond_last_machine | cond_not_machine]

        # templates
        templates = df_templates['template'].to_list()
        original_template = df_templates[df_templates['tid']=='H'].sample(1)['template'].item()
        print("Original template: ", original_template)
        
        df_analyse_full = []
        models = {
            'trained': model,
            'untrained': CausalLanguageModel(
                model_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                fast_tkn=True if not ('opt' in model_name) else False, #because of a bug in OPT
                fp16=args.fp16,
                padding_side='left', untrained=True)
            }
        for (train_mode, this_model) in models.items():
            # prepare to extract kn neurones
            kn_act_buffer = this_model.enable_output_knowledge_neurons()
            # same for extracting mlp and attn output
            mlp_out_buffer = this_model.enable_output_mlp_out()
            attn_out_buffer = this_model.enable_output_attn_out()
            for j in range(5):
                # randomly choose a subject/object pair. Then, for each template, print the per layer intermediate prediction
                test_pair = lama_samples[j]
                token_gold = this_model.tokenizer.encode(test_pair[1])[0]
                print(f'\n[{test_pair[0]}, {test_pair[1]}]')

                """
                Extract intermediate predictions
                """
                original_intermediate_preds, original_intermediate_probs, original_intermediate_reps, original_activations, original_mlp, original_mlp_pred, original_attn, original_attn_pred = forward_extract_from_pair(this_model, [original_template, ], test_pair, kn_act_buffer)
                all_intermediate_preds, all_intermediate_probs, all_intermediate_reps, all_activations, all_mlp, all_mlp_pred, all_attn, all_attn_pred = forward_extract_from_pair(this_model, templates, test_pair, kn_act_buffer)
                df_analyse_pair = []
                for layer in range(this_model.get_nb_layers()): 
                    df_analyse = df_templates.copy()
                    df_analyse['target'] = f'{test_pair[0]}_{test_pair[1]}'
                    df_analyse['layer'] = [layer,] * len(df_analyse)
                    df_analyse['train_mode'] = [train_mode,] * len(df_analyse)
                    # compare with original prompt
                    df_analyse['d_ori_rep'] = [(original_intermediate_reps[0,layer]-all_intermediate_reps[i, layer]).abs().sum().item() for i in range(len(templates))]
                    df_analyse['d_ori_prob'] = [kl_div(torch.log(all_intermediate_probs[i, layer]), original_intermediate_probs[0,layer], log_target=False, reduction="batchmean").item() for i in range(len(templates))]
                    df_analyse['csn_ori_rep'] = [cosine_similarity(all_intermediate_reps[i, layer].float(),original_intermediate_reps[0,layer].float(), dim=-1).item() for i in range(len(templates))]
                    # compare with previous rep
                    df_analyse['d_previous'] = [(all_intermediate_reps[i, layer-1]-all_intermediate_reps[i, layer]).abs().sum().item() for i in range(len(templates))] if layer>0 else [0,]*len(templates)
                    df_analyse['d_previous_prob'] = [kl_div(torch.log(all_intermediate_probs[i, layer-1]),all_intermediate_probs[i, layer], reduction="batchmean").item() for i in range(len(templates))] if layer>0 else [0,]*len(templates)
                    # probabilty dynamic
                    df_analyse['p_gold'] = [all_intermediate_probs[i, layer, token_gold].item() for i in range(len(templates))]
                    df_analyse['p_max'] = [all_intermediate_probs[i, layer].max().item() for i in range(len(templates))]
                    df_analyse['p_pred'] = [all_intermediate_probs[i, layer, torch.argmax(all_intermediate_probs[i, -1])].item() for i in range(len(templates))]
                    # measure the norm
                    df_analyse['h_norm'] = [torch.linalg.norm(all_intermediate_reps[i, layer]).item() for i in range(len(templates))]             
                    # knowledge neurones
                    df_analyse['d_ori_kn'] = [(original_activations[0,layer]-all_activations[i, layer]).abs().sum().item() for i in range(len(templates))]
                    df_analyse['csn_ori_kn'] = [cosine_similarity(original_activations[0,layer].float(),all_activations[i, layer].float(), dim=-1).item() for i in range(len(templates))]
                    df_analyse['norm_kn'] = [torch.linalg.norm(all_activations[i, layer]).item() for i in range(len(templates))]
                    # intermediate pred
                    df_analyse['intermediate_pred'] = [preds.split('\t')[layer] for preds in all_intermediate_preds]
                    # mlp out
                    df_analyse['mlp_pred'] = [preds.split('\t')[layer] for preds in all_mlp_pred]
                    df_analyse['csn_previous_mlp'] = [cosine_similarity(all_mlp[i, layer-1].float(),all_mlp[i, layer].float(), dim=-1).item() for i in range(len(templates))] if layer>0 else [0,]*len(templates)
                    df_analyse['csn_ori_mlp'] = [cosine_similarity(all_mlp[i, layer].float(),original_mlp[0, layer].float(), dim=-1).item() for i in range(len(templates))]
                    # attn out
                    df_analyse['attn_pred'] = [preds.split('\t')[layer] for preds in all_attn_pred]
                    df_analyse['csn_previous_attn'] = [cosine_similarity(all_attn[i, layer-1].float(),all_attn[i, layer].float(), dim=-1).item() for i in range(len(templates))] if layer>0 else [0,]*len(templates)
                    df_analyse['csn_ori_attn'] = [cosine_similarity(all_attn[i, layer].float(),original_attn[0, layer].float(), dim=-1).item() for i in range(len(templates))]
                    df_analyse_pair.append(df_analyse.copy())
                df_analyse_pair = pd.concat(df_analyse_pair)
                df_analyse_pair['instance'] = [test_pair[0], ] * len(df_analyse_pair)
                df_analyse_full.append(df_analyse_pair.copy())
        df_analyse_full = pd.concat(df_analyse_full).reset_index()
        print(df_analyse_full)
        df_analyse_full.to_csv('compare_df.tsv', sep='\t')

        # df_analyse_full = pd.read_csv('compare_df.tsv', sep='\t')\
        """
        Intermediate preds
        """
        text = '' 
        for target in df_analyse_full['target'].unique():
            text += f"\nTarget: {target}"
            for str_template in df_analyse_full['str_template'].unique(): 
                text += f"\n\tTemplate: {str_template}"
                try:
                    this_df = df_analyse_full.query(f'target == "{target}"').query(f'str_template == "{str_template}"').query('train_mode == "trained"')
                    text += '\n\t\t'+repr(','.join(this_df[['layer', 'intermediate_pred']].drop_duplicates().sort_values(by='layer')['intermediate_pred'].to_list()))
                except SyntaxError:
                    text += '\n\t\tSyntaxError'
                    continue
        with open('intermediate_preds.txt', 'w') as f:
            f.write(text)
        """
        MLP preds
        """
        text = '' 
        for target in df_analyse_full['target'].unique():
            text += f"\nTarget: {target}"
            for str_template in df_analyse_full['str_template'].unique(): 
                text += f"\n\tTemplate: {str_template}"
                try:
                    this_df = df_analyse_full.query(f'target == "{target}"').query(f'str_template == "{str_template}"').query('train_mode == "trained"')
                    text += '\n\t\t'+repr(','.join(this_df[['layer', 'mlp_pred']].drop_duplicates().sort_values(by='layer')['mlp_pred'].to_list()))
                except SyntaxError:
                    text += '\n\t\tSyntaxError'
                    continue

        with open('mlp_preds.txt', 'w') as f:
            f.write(text)
        """
        Attn preds
        """
        text = '' 
        for target in df_analyse_full['target'].unique():
            text += f"\nTarget: {target}"
            for str_template in df_analyse_full['str_template'].unique(): 
                text += f"\n\tTemplate: {str_template}"
                try:
                    this_df = df_analyse_full.query(f'target == "{target}"').query(f'str_template == "{str_template}"').query('train_mode == "trained"')
                    text += '\n\t\t'+repr(','.join(this_df[['layer', 'attn_pred']].drop_duplicates().sort_values(by='layer')['attn_pred'].to_list()))
                except SyntaxError:
                    text += '\n\t\tSyntaxError'
                    continue

        with open('attn_preds.txt', 'w') as f:
            f.write(text)
        """
        distance with previous layer MLP
        """
        g = sns.relplot(data=df_analyse_full, x='layer', y='csn_previous_mlp', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'csn_previous_mlp_{model_name_parse}_{relation}.png')
        plt.clf()
        """
        distance with original MLP
        """
        g = sns.relplot(data=df_analyse_full, x='layer', y='csn_ori_mlp', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'csn_ori_mlp_{model_name_parse}_{relation}.png')
        plt.clf()
        """
        distance with previous layer attn
        """
        g = sns.relplot(data=df_analyse_full, x='layer', y='csn_previous_attn', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'csn_previous_attn_{model_name_parse}_{relation}.png')
        plt.clf()
        """
        distance with original attn
        """
        g = sns.relplot(data=df_analyse_full, x='layer', y='csn_ori_attn', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'csn_ori_attn_{model_name_parse}_{relation}.png')
        plt.clf()
        """
        distance with original res
        """
        g = sns.relplot(data=df_analyse_full, x='layer', y='csn_ori_rep', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'csn_ori_rep_{model_name_parse}_{relation}.png')
        plt.clf()
        """
        ppl
        """
        g = sns.barplot(data=df_analyse_full, x='type', y="ppl")
        g.set(yscale="log")
        g.figure.savefig(f'ppl_{model_name_parse}_{relation}.png')
        plt.clf()
        """
        distance between ori anfd mutated in residual stream
        """
        g = sns.relplot(data=df_analyse_full, x='layer', y='d_ori_rep', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'd_ori_rep_{model_name_parse}_{relation}.png')
        plt.clf()
        g = sns.relplot(data=df_analyse_full, x='layer', y='d_ori_prob', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'd_ori_prob_{model_name_parse}_{relation}.png')
        plt.clf()
        """
        distance with previous layer residual stream
        """
        g = sns.relplot(data=df_analyse_full, x='layer', y='d_previous', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'd_prev_rep_{model_name_parse}_{relation}.png')
        plt.clf()
        g = sns.relplot(data=df_analyse_full, x='layer', y='d_previous_prob', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'd_prev_prob_{model_name_parse}_{relation}.png')
        plt.clf()
        """
        Evolution of p_gold
        """
        g = sns.relplot(data=df_analyse_full, x='layer', y='p_gold', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'p_gold_{model_name_parse}_{relation}.png')
        plt.clf()
        g = sns.relplot(data=df_analyse_full, x='layer', y='p_gold', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'p_gold_2_{model_name_parse}_{relation}.png')
        plt.clf()
        g = sns.relplot(data=df_analyse_full, x='layer', y='p_max', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'p_max_{model_name_parse}_{relation}.png')
        plt.clf()
        g = sns.relplot(data=df_analyse_full, x='layer', y='p_pred', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'p_pred_{model_name_parse}_{relation}.png')
        plt.clf()
        """
        Evolution of the residual stream's norm
        """
        g = sns.relplot(data=df_analyse_full, x='layer', y='h_norm', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'h_norm_{model_name_parse}_{relation}.png')
        plt.clf()
        """
        knowledge neurones
        """
        g = sns.relplot(data=df_analyse_full, x='layer', y='norm_kn', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'norm_kn_{model_name_parse}_{relation}.png')
        plt.clf()
        g = sns.relplot(data=df_analyse_full, x='layer', y='d_ori_kn', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'd_ori_kn_{model_name_parse}_{relation}.png')
        plt.clf()
        g = sns.relplot(data=df_analyse_full, x='layer', y='csn_ori_kn', hue='type', col="train_mode", kind="line")#, style='template')
        g.set(yscale="log")
        g.figure.savefig(f'csn_ori_kn_{model_name_parse}_{relation}.png')
        plt.clf()
        exit()
        

        # for layer in range(all_intermediate_representation.size(1)): 
        #     fig, ax = plt.subplots()   
        #     """
        #      compute distance heatmap: residual stream
        #     """
        #     this_layer_intermediate_representation = all_intermediate_representation[:, layer]
        #     distance_intermediate_rep = [torch.stack([(this_layer_intermediate_representation[i_a]-this_layer_intermediate_representation[i_b]).abs().sum() for i_a in range(len(filled_templates))]) for i_b in range(len(filled_templates))]
        #     distance_intermediate_rep = torch.stack(distance_intermediate_rep).numpy()
        #     print(distance_intermediate_rep)

            
        #     # using the variable axs for multiple Axes
        #     im = ax.imshow(distance_intermediate_rep)

        #     # Show all ticks and label them with the respective list entries
        #     ax.set_xticks(np.arange(len(filled_templates)), labels=filled_templates)
        #     ax.set_yticks(np.arange(len(filled_templates)), labels=filled_templates)

        #     # Rotate the tick labels and set their alignment.
        #     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #             rotation_mode="anchor")

        #     # Loop over data dimensions and create text annotations.
        #     for i in range(len(filled_templates)):
        #         for j in range(len(filled_templates)):
        #             text = ax.text(j, i, distance_intermediate_rep[i, j],
        #                         ha="center", va="center", color="w")

        #     ax.set_title("Distance in intermediate representation between prompts")
        #     fig.tight_layout()
        #     plt.savefig(f'inter_rep_dist_{layer}_{model_name_parse}_{relation}_{test_pair[0]}.png')

        #     plt.clf()

        # """
        # Intermediate mlp out
        # """
        # all_intermediate_mlp_out = []
        # for i in range(len(filled_templates)):
        #     text_input = filled_templates[i]
        #     print('\n--- ', text_input)

        #     output = model.forward_pass_nograd(text_input, tokenize=True)
        #     mlp_out = [mlp_out_buffer[l].detach().clone() for l in range(model.get_nb_layers())] # updated after each forward pass

        #     all_intermediate_mlp_out.append(torch.stack([v[-1] for v in mlp_out])) # only keep representation of the last token
        # all_intermediate_mlp_out = torch.stack(all_intermediate_mlp_out)

        # for layer in range(all_intermediate_mlp_out.size(1)): 
        #     fig, ax = plt.subplots()
        #     """
        #      compute distance heatmap: residual stream
        #     """
        #     this_layer_mlp_out = all_intermediate_mlp_out[:, layer]
        #     distance_mlp_out = [torch.stack([(this_layer_mlp_out[i_a]-this_layer_mlp_out[i_b]).abs().sum() for i_a in range(len(filled_templates))]) for i_b in range(len(filled_templates))]
        #     distance_mlp_out = torch.stack(distance_mlp_out).numpy()
        #     print(distance_mlp_out)

            
        #     # using the variable axs for multiple Axes
        #     im = ax.imshow(distance_mlp_out)

        #     # Show all ticks and label them with the respective list entries
        #     ax.set_xticks(np.arange(len(filled_templates)), labels=filled_templates)
        #     ax.set_yticks(np.arange(len(filled_templates)), labels=filled_templates)

        #     # Rotate the tick labels and set their alignment.
        #     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #             rotation_mode="anchor")

        #     # Loop over data dimensions and create text annotations.
        #     for i in range(len(filled_templates)):
        #         for j in range(len(filled_templates)):
        #             text = ax.text(j, i, distance_mlp_out[i, j],
        #                         ha="center", va="center", color="w")

            
        #     fig.tight_layout()
        #     ax.set_title("Distance in intermediate mlp out between prompts")
        #     plt.savefig(f'inter_mlpout_dist_{layer}_{model_name_parse}_{relation}_{test_pair[0]}.png')
        #     plt.clf()

        



        # """
        # Topk knowledge neurones
        # """
        # for i in range(len(filled_templates)):
        #     text_input = filled_templates[i]
        #     print('\n--- ', text_input)

        #     output = model.forward_pass_nograd(text_input, tokenize=True)
        #     activations = [kn_act_buffer[l].detach().clone() for l in range(model.get_nb_layers())] # updated after each forward pass

        #     if 'opt' in args.model_name:
        #         for l in range(model.get_nb_layers()):
        #             d1, d2 = activations[l].shape
        #             activations[l] = activations[l].reshape(1, -1, d2).squeeze()
        #             # print(activations[l])
        #             # print(activations[l].shape)
        #             activation_pred = activations[l][-1]
        #             top_kn = torch.topk(activation_pred.float(), 5, largest=True, sorted=True)
        #             print(top_kn.indices.tolist())
            

        # """
        # Compare inter
        # """
        
    exit()

    # initialise the algo
    autoprompt = DiscreteGradientPromptSearch(model, args.n_population, args.num_candidates, n_rounds=3)

    for relation in relations: # in the future, run all relation in parallel in different scrips
        initial_template = paraphrases[relation]
        """
        dataset is a list of tuple [(X,Y), ...]
        where X is used to fill in the template and Y is the expected next token.
        """
        savepath = os.path.join(args.output,f'disc-prompt-search_{model_name_parse}_{relation}_{random_seed}.tsv') 
        autoprompt.train(initial_template, lamaset, relation, args.n_iterations_max, args.batch_size, savepath)
        # dev set
