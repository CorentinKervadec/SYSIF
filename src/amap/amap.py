import sys
sys.path.append('../')
import torch
import logging
from datasets import Dataset
from src.model.causal_lm import CausalLanguageModel
from src.data.dataset_loader import tokenize_slice_batch, load_hf_dataset_with_sampling

from tqdm import tqdm

class LMamap:
    def __init__(self, model: CausalLanguageModel, mode=['input', 'output'], device='cuda', fp16=True) -> None:
        # amap
        self.amap = None
        self.tokens_count = None

        # options
        self.mode = mode
        self.special_tracking = []

        # model: hugging face model
        self.model = model
        self.kn_act_buffer = model.enable_output_knowledge_neurons()
        self.vocab_list = model.get_vocab()
        self.n_layers = model.get_nb_layers()
        self.n_units = model.get_nb_knowledge_neurons(layer_id=0)

        # tensor format
        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32

        # prepare amap
        self.prepare_amap()

    def prepare_amap(self) -> None:
        """
        Prepare the amap
        """

        self.amap = {}
        vocab_size = len(self.vocab_list) 
        amap_dim = (vocab_size, self.n_units)
        
        # init token/unit matrix
        '''
        amap['input'][l][i][j]
        provides the accumulated activation of knowledge neuron j when vocab item i is the last token of the input
        '''
        if 'input' in self.mode:
            self.amap['input'] = [torch.full(size=amap_dim, fill_value=0.0, dtype=self.dtype) for l in range(self.n_layers)]
        if 'output' in self.mode:
            self.amap['output'] = [torch.full(size=amap_dim, fill_value=0.0, dtype=self.dtype) for l in range(self.n_layers)]
        self.tokens_count = {
            'input':torch.zeros(size=(vocab_size,)).int().to(self.device), 
            'output':torch.zeros(size=(vocab_size,)).int().to(self.device)}
    
    def reset_amap(self) -> None:
        # amap
        self.amap = None
        self.tokens_count = None
        # prepare amap
        self.prepare_amap()
        for special in self.special_tracking:
            if special == 'position':
                logging.warning(f'[amap] You also have to call add_position(window_size) again!')
        
    def add_position(self, window_size) -> None:
        logging.warning(f'[amap] Adding position tracking. Make sure that the window size is {window_size} during the amap extraction.')
        amap_pos_dim = (window_size+1, self.n_units) # window_size +1, because we also count the last next token predicted by the LM
        for mode in self.amap:
            for l in range(len(self.amap[mode])):#layers
                amap_pos = torch.full(size=amap_pos_dim, fill_value=0.0, dtype=self.dtype)
                # Concat original amap and amap_pos
                self.amap[mode][l] = torch.cat((self.amap[mode][l], amap_pos), dim=0)
            self.tokens_count[mode] = torch.cat((self.tokens_count[mode], torch.zeros(size=(window_size+1,)).int().to(self.device)))
        self.position_offset = len(self.vocab_list) # use an offset to differenciate token id from token position
                                                    # position i will be refered using the id: i+offset
        if 'position' not in self.special_tracking: self.special_tracking.append('position')


    def update_token_unit(self, unit_tokens, kn_act, layer, unique_id, token_ids, tokens_count, old_token_count):
        '''
        unit_tokens: Table [n_layer, n_tokens, n_kn] where the accummulated value of token/activation pairs is stored. 
        kn_act: knowledge neuron activations
        layer: id of the layer
        unique_id: set of the tokens id in the input (resp. output) sequence (only unique tokens)
        tokens_id: list of the tokens id the input (resp. output) sequence (one token can appear many times)
        tokens_count: count the number of time each token appeared
        old_token_count: previous tokens_count, used to compute the cumulative average
        device: which device to use
        In order to reduce computation, only updates the tokens in unique_id (i.e. those in the input sequence)
        '''
        save_device = unit_tokens[layer].device
        kn_d1, kn_d2, kn_d3 = kn_act[layer].shape
        
        # create an index mask, to only process tokens in the batch. [n_tokens_in_the_input, n_unique_tokens_in_the_input]
        expand_unique_id = unique_id.unsqueeze(0).expand(token_ids.size(0), -1)
        # in the mask, line i correspond to the location of unique token i in the input (resp. output) sequence
        index_mask = (expand_unique_id == token_ids.unsqueeze(-1)).t().type(self.dtype) 
        # compute the unit-token activations for the batch, on the device
        """
        index_mask: [n_uniques_in_sequence, n_tokens_in_sequence]. Line i has 1 where token i appear in the sequence
        kn_act[layer].view(kn_d1*kn_d2, kn_d3): [n_tokens_in_sequence, n_feats]

        batch_unit_token_cum: index_mask x kn_act[layer] = [n_uniques_in_sequence, n_feats].
        Line i correspond to the accumulated kn features obtained when token i is the input.
        """
        batch_unit_token_cum = torch.matmul(index_mask.to(self.device), kn_act[layer].to(self.device).view(kn_d1*kn_d2, kn_d3).type(self.dtype))
        # check nan and inf
        # if batch_unit_token_cum.isnan().any():
        #     logging.warning('[accumulated kn] nan detected!')
        # if batch_unit_token_cum.isinf().any():
        #     logging.warning('[accumulated kn] inf detected!')   
            
        # update the cumuluative average
        unit_tokens[layer][unique_id] = cumulative_average(
            new_item    = batch_unit_token_cum,
            new_count   = tokens_count[unique_id].unsqueeze(-1),
            old_count   = old_token_count[unique_id].unsqueeze(-1),
            old_average = unit_tokens[layer][unique_id.to(save_device)],
            device      = self.device,
        ).to(save_device)

        return unit_tokens


    def extract(self, dataset: Dataset, batch_size, window_size, window_stride):
        
        # preprocess dataset
        dataset_sliced_batched, n_batch = tokenize_slice_batch(dataset, self.model.tokenizer, batch_size, window_size, window_stride, drop_last=True)
        n_sentences = 0

        # iterate on the dataset
        for input_ids, attention_mask in tqdm(dataset_sliced_batched, total=n_batch):
            d_batch, d_sent = input_ids.shape
            n_sentences += len(input_ids)
            
            # forward pass
            output = self.model.forward_pass_nograd((input_ids, attention_mask), tokenize=False)
            
            # accumulate input and output token ids
            tokens_ids = {
                'input': input_ids.flatten().to(self.device).type(self.dtype),
                'output': torch.argmax(output.logits.detach(), dim=-1).flatten().to(self.device).type(self.dtype)
            }
            # count unique tokens
            unique_id = {}
            old_token_count = {}

            # get activations
            activations = self.kn_act_buffer

            for mode in self.amap.keys():
                """
                Detect the unique tokens_id in the input sequence, and count them.
                Update the total tokens_count.
                Save an old_tokens_count, used to compute the cumulative average
                """
                
                ids = tokens_ids[mode] # list of ids that we will be used to update the amap
                                       # corresponds to the token id + the various additional attributes that are tracked

                for special in self.special_tracking:
                    if special == 'position':
                        """
                        The token position to the list of ids
                        """
                        if mode == 'input':
                            position_id = torch.arange(window_size).unsqueeze(0).expand((batch_size, -1)) + self.position_offset
                        elif mode == 'output': # same as input, but +1 because LM output the next token
                            position_id = torch.arange(window_size).unsqueeze(0).expand((batch_size, -1)) + self.position_offset + 1
                        ids = torch.cat((ids, position_id.to(ids.device).flatten()), dim=0) # add the position id to the token_id
                        # we also have to duplicate the activation matrix
                        for l in range(self.n_layers):
                            activations[l] = torch.cat((activations[l], activations[l]), dim=0)

                unique_tokens_with_count = torch.unique(ids, return_counts=True)
                unique_id = unique_tokens_with_count[0].long() # set of unique tokens in the input (resp. output)
                count_id = unique_tokens_with_count[1] # corresponding count for each unique token

                old_token_count = self.tokens_count[mode].clone().detach() # save old count
                self.tokens_count[mode][unique_id] += count_id # update new count                       

                # for each layer accumulate the unit-token association
                for l in range(self.n_layers):
                    # per token stats
                    with torch.no_grad():
                        self.amap[mode] = self.update_token_unit(
                            unit_tokens=self.amap[mode],
                            kn_act=self.kn_act_buffer,
                            layer=l,
                            unique_id=unique_id,
                            token_ids=ids,
                            tokens_count=self.tokens_count[mode],
                            old_token_count=old_token_count,)

        for mode in self.amap.keys():  
            self.tokens_count[mode] = self.tokens_count[mode].cpu()
            for l in range(self.n_layers):
                # check nan and inf
                if self.amap[mode][l].isnan().any():
                    logging.warning(f'[Finished][{mode}][{l}] nan detected!')
                if self.amap[mode][l].isinf().any():
                    logging.warning(f'[Finished][{mode}][{l}] inf detected!')  
                self.amap[mode][l] = self.amap[mode][l].cpu()

        return self.amap, self.tokens_count, n_sentences
    
    def sanity_check(self, n_sentences):
        warning_flag = ''
        if 'input' in self.mode and 'output' in self.mode:
            try:
                input_token_sum = self.tokens_count['input'].sum(0)
                output_token_sum = self.tokens_count['output'].sum(0)
                assert input_token_sum == output_token_sum
            except AssertionError:
                logging.warning(f'Input and output do not have the same number of tokens! Input:{input_token_sum}. Output:{output_token_sum}')
                warning_flag += 'wTkn'
            try:
                total_input = (self.amap['output'][0].float() * self.tokens_count['output'].unsqueeze(-1).float()).sum()
                total_output = (self.amap['output'][0].float() * self.tokens_count['output'].unsqueeze(-1).float()).sum()
                assert abs(total_input - total_output) < 1
            except AssertionError:
                logging.warning(f'Input and output do not have the same total activation! Input:{total_input}. Output:{total_output}')
                warning_flag += 'wAct'
        if 'position' in self.special_tracking:
            # check that position 0 is counted n_sample times
            try:
                position_count = self.tokens_count['input'][0+self.position_offset]
                assert position_count == n_sentences
            except AssertionError:
                logging.warning(f'Position 0 count is not correct! Position 0 count:{position_count}. Number of sentences:{n_sentences}')
                warning_flag += 'wPos'
        return warning_flag

def cumulative_average(new_item, new_count, old_count, old_average, device='cpu'):
    datatype = old_average.dtype
    new_item = new_item.to(device).type(torch.float32)
    new_count = new_count.to(device).type(torch.float32)
    old_count = old_count.to(device).type(torch.float32)
    old_average = old_average.to(device).type(torch.float32)
    old_sum = old_count * old_average # float32, to avoid overflow
    cum_avg = (new_item + (old_count) * old_average) / (new_count)
    # check nan and inf
    # if cum_avg.isnan().any():
    #     logging.warning('[cumulative average f32] nan detected!')
    # if cum_avg.isinf().any():
    #     logging.warning('[cumulative average f32] inf detected!')    
    cum_avg = cum_avg.type(datatype)
    # check nan and inf
    # if cum_avg.isnan().any():
    #     logging.warning('[cumulative average after cast] nan detected!')
    # if cum_avg.isinf().any():
    #     logging.warning('[cumulative average after cast] inf detected!')    
    return cum_avg