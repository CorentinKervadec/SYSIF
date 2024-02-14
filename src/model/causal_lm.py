import torch
import torch.nn.functional as F
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Callable

HF_TOKEN_LLAMA2='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'

class CausalLanguageModel:
    def __init__(self, model_name, device="cpu", fast_tkn=True, fp16=True, padding_side='right', untrained=False):
        self.device = torch.device(device)
        self.model_name = model_name
        self.tokenizer = self.prepare_tokenizer(model_name, fast_tkn, padding_side=padding_side)
        if untrained:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16 if fp16 else torch.float32).to(self.device)
        else:
            if 'llama' in model_name:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code="true",
                    torch_dtype=torch.float16 if fp16 else torch.float32,
                    use_auth_token=HF_TOKEN_LLAMA2
                ).to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if fp16 else torch.float32).to(self.device)
        # print(self.model)
        self.layer_act = self.get_act_fn()

    def generate_tokens_batch(self, input_ids, attention_mask, n_tokens):
        # require left padding
        tokens_generated = self.model.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            max_new_tokens=n_tokens, do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id)
        tokens_generated = tokens_generated[:, -n_tokens:]
        return tokens_generated

    def generate_tokens_from_text_batch(self, text_input, n_tokens):
        tokenized = self.tokenizer(text_input, padding=True, return_tensors='pt').to(self.device)
        tokens_generated = self.generate_tokens_batch(tokenized.input_ids, tokenized.attention_mask, n_tokens)
        text_generated = [self.tokenizer.decode(t) for t in tokens_generated]
        return text_generated

    # def generate_text(self, prompt, max_length=50, num_return_sequences=1):
    #     input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
    #     output = self.model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences, no_repeat_ngram_size=2)
    #     generated_text = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
    #     return generated_text

    def forward_per_layer(self, inputs):
        n_layers = self.get_nb_layers()
        for l_idx in range(n_layers):
            # process
            print('')

        return None

    def forward_pass_from_text(self, input_text):
        input_ids = self.tokenizer(input_text, padding=True, return_tensors='pt').to(self.device)
        output = self.model(**input_ids)
        return (output, input_ids['attention_mask'])
    
    def forward_pass_from_tkns(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask)
        return output

    def forward_pass_nograd(self, input, tokenize=True):
        with torch.no_grad():
            output = self.forward_pass(input, tokenize)    
        return output
    
    def forward_pass(self, input, tokenize=True):
        if tokenize:
            output = self.forward_pass_from_text(input)
        else:
            input_ids, attention_mask = input
            output = self.forward_pass_from_tkns(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
        return output

    def get_output_probability(self, input_text):
        output = self.forward_pass_nograd(input_text)
        logits = output.logits
        probabilities = logits.softmax(dim=-1)
        return probabilities

    def get_intermediate_mlp_output(self, input_text, stride=2):
        """
        does not work with batching
        """
        all_intermediate_out = []
        unembed = self.get_unembed()
        self.enable_output_hidden_states()
        output = self.forward_pass_nograd(input_text)[0]
        n_layers = self.get_nb_layers()
        iterator_list = [l_idx for l_idx in range(0,n_layers,stride)]
        if iterator_list[-1] != (n_layers-1): # make sure that the last layer is considered
            iterator_list.append(n_layers-1)
        for l_idx in range(0,n_layers,stride):
            intermediate_output = output.hidden_states[l_idx+1].squeeze()[-1].detach() # +1 because the first is the embedding / squeeze because we are not batching
            intermediate_logit = unembed(intermediate_output)
            intermediate_argmax = torch.argmax(intermediate_logit, dim=-1)
            # intermediate_prob = torch.softmax(intermediate_logit, dim=-1)
            intermediate_pred = self.tokenizer.decode(intermediate_argmax)
            all_intermediate_pred.append(intermediate_pred)
            all_intermediate_rep.append(intermediate_output.cpu())
        all_intermediate_pred = '\t'.join(all_intermediate_pred)
        all_intermediate_rep = torch.stack(all_intermediate_rep)
        return all_intermediate_out


    def get_intermediate_output(self, input_text, stride=1, tokenize=False):
        """
        does not work with batching
        """
        all_intermediate_pred = []
        all_intermediate_rep = []
        all_intermediate_prob = []
        all_activations = []
        # MLP
        all_mlp = []
        all_mlp_pred = []
        # ATTN
        all_attn = []
        all_attn_pred = []
        #
        unembed = self.get_unembed()
        self.enable_output_hidden_states()
        if tokenize:
            output = self.forward_pass_nograd(input_text)[0]
        else:
            output = self.forward_pass_nograd(input_text, tokenize=False)
        n_layers = self.get_nb_layers()
        iterator_list = [l_idx for l_idx in range(0,n_layers,stride)]
        if iterator_list[-1] != (n_layers-1): # make sure that the last layer is considered
            iterator_list.append(n_layers-1)
        for l_idx in range(0,n_layers,stride):
            intermediate_output = output.hidden_states[l_idx+1].squeeze()[-1].detach() # +1 because the first is the embedding / squeeze because we are not batching
            intermediate_logit = unembed(intermediate_output)
            intermediate_argmax = torch.argmax(intermediate_logit, dim=-1)
            intermediate_prob = torch.softmax(intermediate_logit, dim=-1)
            try:
                intermediate_pred = self.tokenizer.decode(intermediate_argmax)
            except TypeError:
                print("TYPE ERROR, intermediate pred:")
                print('intermediate_argmax: ', intermediate_argmax)
                print('intermediate_logit: ', intermediate_logit)
                intermediate_pred = '--NoneType--'
            
            all_intermediate_pred.append(intermediate_pred)
            all_intermediate_rep.append(intermediate_output.cpu())
            all_intermediate_prob.append(intermediate_prob.cpu())
            all_activations.append(self.kn_act_buffer[l_idx][-1].detach().clone())
            # mlp
            mlp_out = self.mlp_out_buffer[l_idx][-1].detach().clone()
            mlp_logit = unembed(mlp_out.to(self.device))
            mlp_argmax = torch.argmax(mlp_logit, dim=-1).cpu()
            try:
                mlp_pred = self.tokenizer.decode(mlp_argmax)
            except TypeError:
                print("TYPE ERROR, mlp pred:")
                print('mlp_argmax: ', mlp_argmax)
                print('mlp_logit: ', mlp_logit)
                mlp_pred = '--NoneType--'
            all_mlp.append(mlp_out)
            all_mlp_pred.append(mlp_pred)
            # attn
            attn_out = self.attn_out_buffer[l_idx].squeeze()[-1].detach().clone() # squeeze works because we have batch size of one
            attn_logit = unembed(attn_out.to(self.device))
            attn_argmax = torch.argmax(attn_logit, dim=-1).cpu()
            try:
                attn_pred = self.tokenizer.decode(attn_argmax)
            except TypeError:
                print("TYPE ERROR, attn pred:")
                print('attn_argmax: ', attn_argmax)
                print('attn_logit: ', mlp_logit)
                mlp_pred = '--NoneType--'
            all_attn.append(attn_out)
            all_attn_pred.append(attn_pred)
        all_intermediate_pred = '\t'.join(all_intermediate_pred)
        all_intermediate_rep = torch.stack(all_intermediate_rep)
        all_intermediate_prob = torch.stack(all_intermediate_prob)
        all_activations = torch.stack(all_activations)
        # mlp
        all_mlp = torch.stack(all_mlp)
        all_mlp_pred = '\t'.join(all_mlp_pred)
        # attn
        all_attn = torch.stack(all_attn)
        all_attn_pred = '\t'.join(all_attn_pred)
        return all_intermediate_pred, all_intermediate_prob, all_intermediate_rep, all_activations, all_mlp, all_mlp_pred, all_attn, all_attn_pred

    def compute_ppl_from_tokens_batch(self, input_ids, attention_mask):
        if attention_mask is None:
            # pad and create the attention mask
            max_length = max([len(t) for t in input_ids])
            input_ids = torch.stack([F.pad(torch.tensor(t), (max_length-len(t),0), value=self.tokenizer.pad_token_id) for t in input_ids])
            attention_mask = torch.where(input_ids.eq(self.tokenizer.pad_token_id),0,1)
        
        target_ids = input_ids.clone()
        target_ids = torch.where(target_ids==self.tokenizer.pad_token_id, 0, target_ids) # ignore padding tokens
        # shift targets, as each position predicts the next token
        target_ids = shift_batch_tensor(target_ids).to(self.device)
        mask = 1 * target_ids.eq(0) # 1 means the position has to be masked
        mask[:, -1] = 1 # do not compute the loss on the last token as we don't know what is the next token
        with torch.no_grad():
            outputs = self.model(input_ids.to(self.device), attention_mask.to(self.device))
            logits = outputs.logits
            loss = nll(logits, target_ids)
            # mask padding tokens + average over valid tokens
            loss = (loss * (1-mask)).sum(-1) / (1-mask).sum(-1)
        ppl = torch.exp(loss.float()).to('cpu')        
        return ppl

    def compute_ppl_from_text_batch(self, input_text):
        encodings = self.tokenizer(input_text, padding=True, return_tensors='pt').to(self.device)
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask
        ppl = self.compute_ppl_from_tokens_batch(input_ids, attention_mask)
        return ppl

    def enable_output_hidden_states(self):
        self.model.config.output_hidden_states=True

    def enable_output_attention_maps(self):
        self.model.config.output_attentions=True

    def enable_output_knowledge_neurons(self):
        # define the tensor where the activation will be stored
        layers = self.get_layers()
        n_layers = len(layers)
        self.kn_act_buffer={lid: torch.empty(0) for lid in range(n_layers)}
        # Setting a hook for saving FFN intermediate output
        for lid, layer in enumerate(layers):
            self.get_knowledge_neurons(lid).register_forward_hook(self.save_kn_act_hook(lid))
        return self.kn_act_buffer
    
    def enable_output_mlp_out(self):
        # define the tensor where the activation will be stored
        layers = self.get_layers()
        n_layers = len(layers)
        self.mlp_out_buffer={lid: torch.empty(0) for lid in range(n_layers)}
        # Setting a hook for saving FFN intermediate output
        for lid, layer in enumerate(layers):
            self.get_mlp_out(lid).register_forward_hook(self.save_mlp_out_hook(lid, activation=False))
        return self.mlp_out_buffer

    def save_mlp_out_hook(self, layer, activation=True) -> Callable:
        def fn(_, __, output):
            before_act = output.detach()
            # print("mlp shape: ", before_act.shape)
            # shape is probably [batch*seq_len, h_dim]
            if activation:
                after_act = self.layer_act(before_act) # apply activation
                self.mlp_out_buffer[layer] = after_act.cpu()
            else:
                self.mlp_out_buffer[layer] = before_act.cpu()
        return fn   

    def enable_output_attn_out(self):
        # define the tensor where the activation will be stored
        layers = self.get_layers()
        n_layers = len(layers)
        self.attn_out_buffer={lid: torch.empty(0) for lid in range(n_layers)}
        # Setting a hook for saving FFN intermediate output
        for lid, layer in enumerate(layers):
            self.get_attn_out(lid).register_forward_hook(self.save_attn_out_hook(lid, activation=False))
        return self.mlp_out_buffer

    def save_attn_out_hook(self, layer, activation=True) -> Callable:
        def fn(_, __, output):
            before_act = output[0].detach()
            # print("attn shape: ", before_act.shape)
            # shape is probably [batch, seq_len, h_dim]. Different than mlp
            # todo: check the shape of before act
            if activation:
                after_act = self.layer_act(before_act) # apply activation
                self.attn_out_buffer[layer] = after_act.cpu()
            else:
                self.attn_out_buffer[layer] = before_act.cpu()
        return fn

    def save_kn_act_hook(self, layer, activation=True) -> Callable:
        def fn(_, __, output):
            before_act = output.detach()
            if activation:
                after_act = self.layer_act(before_act) # apply activation
                self.kn_act_buffer[layer] = after_act.cpu()
            else:
                self.kn_act_buffer[layer] = before_act.cpu()
        return fn   

    # def get_attention_maps(self, input_text):
    #     input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
    #     with torch.no_grad():
    #         output = self.model(input_ids)
    #         attentions = output.attentions
    #     return attentions

    # def get_activation_values(self, input_text):
    #     input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
    #     with torch.no_grad():
    #         output = self.model(input_ids)
    #         hidden_states = output.hidden_states
    #     return hidden_states

    # def get_intermediate_representation(self, input_text, layer_id):
    #     input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
    #     with torch.no_grad():
    #         output = self.model(input_ids)
    #         intermediate_layer = output.hidden_states[layer_id]
    #     return intermediate_layer 
    
    def get_vocab(self):
        vocab = [self.tokenizer.decode(i) for i in range(len(self.tokenizer))]
        # if 'opt' in self.model_name:
        #     vocab = list(self.tokenizer.encoder)
        # elif 'pythia' in self.model_name:
        #     vocab = self.tokenizer.vocab
        # elif 'mistral' in self.model_name:
        #     vocab = list(self.tokenizer.vocab)
        # else:
        #     vocab = self.tokenizer.vocab
        return vocab
    
    def get_vocab_size(self):
        # warning: tokenizer.vocab_size only return the vocab size without taking into account the added tokens.
        return len(self.tokenizer)

    def get_embeddings(self):
        if 'opt' in self.model_name:
            embeddings = self.model.model.decoder.embed_tokens
        elif 'pythia' in self.model_name:
            embeddings = self.model.gpt_neox.embed_in
        elif 'mistral' in self.model_name:
            embeddings = self.model.model.embed_tokens
        else:
            embeddings = self.model.gpt_neox.embed_in
        return embeddings

    def get_layers(self):
        if 'opt' in self.model_name:
            layers = self.model.model.decoder.layers
        elif 'pythia' in self.model_name:
            layers = self.model.gpt_neox.layers 
        elif 'mistral' in self.model_name:
            layers = self.model.model.layers 
        else:
            layers = self.model.gpt_neox.layers
        return layers

    def get_nb_layers(self):
        return len(self.get_layers())
    
    def get_knowledge_neurons(self, layer_id):
        layers = self.get_layers()
        if 'opt' in self.model_name:
            kn = layers[layer_id].fc1
        elif 'pythia' in self.model_name:
            kn = layers[layer_id].mlp.dense_h_to_4h
        else:
            kn = layers[layer_id].fc1
        return kn
    
    def get_mlp_out(self, layer_id):
        layers = self.get_layers()
        if 'opt' in self.model_name:
            fc2 = layers[layer_id].fc2
        elif 'pythia' in self.model_name:
            fc2 = None # TODO layers[layer_id].mlp.dense_h_to_4h
        else:
            fc2 = layers[layer_id].fc2
        return fc2
    
    def get_attn_out(self, layer_id):
        layers = self.get_layers()
        if 'opt' in self.model_name:
            fc2 = layers[layer_id].self_attn
        elif 'pythia' in self.model_name:
            fc2 = None # TODO layers[layer_id].mlp.dense_h_to_4h
        else:
            fc2 = layers[layer_id].fc2
        return fc2
      
    def get_nb_knowledge_neurons(self, layer_id=None):
        if layer_id is not None:
            return self.get_knowledge_neurons(layer_id).out_features
        else:
            return sum([self.get_knowledge_neurons(i).out_features for i in range(self.get_nb_layers())])

    def get_act_fn(self):
        if 'opt' in self.model_name:
            act_str = self.model.model.config.activation_function
        elif 'pythia' in self.model_name:
            act_str = self.model.config.hidden_act
        elif 'mistral' in self.model_name:
            act_str = self.model.config.hidden_act
        else: # default
            act_str = 'relu'
        
        if act_str == 'relu':
            return torch.nn.ReLU()
        elif act_str == 'gelu':
            return torch.nn.GELU()
        elif act_str == 'silu':
            return torch.nn.SiLU()
        else: # relu by default
            return torch.nn.GELU()

    def get_unembed(self):
        if 'opt' in self.model_name:
            unembed = self.model.lm_head
        elif 'pythia' in self.model_name:
            act_str = self.model.embed_out
        elif 'mistral' in self.model_name:
            #todo
            return None
        else: # default
            return None
        return unembed

    def prepare_tokenizer(self, model_name, fast_tkn, padding_side):
        if 'llama' in model_name:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                use_fast=fast_tkn,
                # legacy=True,
                token=HF_TOKEN_LLAMA2,
                padding_side=padding_side
            )
            # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.3b")
            # bos_id = tokenizer.bos_token_id
            # tokenizer.pad_token_id = bos_id
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=fast_tkn, padding_side=padding_side)
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        return tokenizer

    def prompt(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        tokens = self.model.generate(**inputs.to(self.device))
        return self.tokenizer.decode(tokens[0])

def nll(logits, label_ids):
    label_ids = label_ids.unsqueeze(-1)
    predict_logp = F.log_softmax(logits, dim=-1)
    target_logp = predict_logp.gather(-1, label_ids)
    return -target_logp.squeeze()

def shift_batch_tensor(t):
    return torch.concat([t[:,1:], t[:,0].unsqueeze(-1)], dim=1)

if __name__ == "__main__":
    # Example usage
    model_name = "gpt2"  # You can replace this with the model name you want to use
    lm = CausalLanguageModel(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    print(lm)