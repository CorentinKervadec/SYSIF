import sys
sys.path.append('../')
import logging
from datasets import load_dataset, Dataset
import random
import torch
import torch.nn.functional as F
import itertools


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_hf_dataset_with_sampling(name, n_samples=None):
    """
    Load a dataset from Hugging Face and optionally sample a subset of it.

    Args:
        name (str): Name of the dataset to load in the format "path,name,split"
        n_samples (int or None): Number of samples to randomly select from the dataset. If None, load the full dataset.

    Returns:
        datasets.Dataset: The loaded dataset.
    """
    try:
        path, name, split = name.split(',')
        dataset = load_dataset(path, name, split=split)
        if n_samples is not None and n_samples > 0:
            random_idx = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
            dataset = dataset.select(random_idx)
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset '{name}' from Hugging Face: {str(e)}")
        raise ValueError(f"Failed to load dataset '{name}' from Hugging Face: {str(e)}")
    

def slice_tokenized_datum(tokenized_datum, window_size, window_stride):
    datum_length = len(tokenized_datum)
    start_offset = random.randint(0,window_size-1) # add a random start offset to increase the diversity (and avoid having too much bos)
    slices = [tokenized_datum[t:t+window_size] for t in range(start_offset,datum_length-window_size,window_stride)]
    return slices



def batchify(datalist, batch_size, drop_last, tokenizer=None, output_text=False, pad=None):
    """
    /!\ if you don't specify a tokenizer, it assumes that the input
    is already tokenized and contains only samples of the same length (unless you set output_text=True)
    If the input is already tokenized but you want to pad it to the same length, use tokenizer=None and pad=self.model.tokenizer.pad_token_id
    (otherwise, set pad to none)
    """
    batches = [datalist[i:i+batch_size] for i in range(0,len(datalist),batch_size)]

    # drop last to avoid different batch size
    if drop_last:
        if not len(batches[-1]) == batch_size:
            batches = batches[:-1]

    n_batches = len(batches)

    # tokenize
    if tokenizer is not None:
        batches = [tokenizer(batch, padding=True, return_tensors="pt") for batch in batches]
    elif not output_text:
        if pad is not None:
            # tokenize and (left) pad the inputs
            pad_batches = []
            for this_batch in batches:
                max_length = max([len(t) for t in this_batch])
                inputs = torch.stack([F.pad(torch.tensor(t), (max_length-len(t), 0), value=pad) for t in this_batch])
                attention_masks = torch.where(inputs.eq(pad),0,1)
                pad_batches.append((inputs, attention_masks))
            batches = pad_batches
        else:    
            # to pytorch
            batches = [torch.tensor(batch) for batch in batches]
            attention_masks = [torch.ones_like(batch) for batch in batches]
            batches = zip(batches, attention_masks)

    return batches, n_batches

def tokenize_slice_batch(dataset: Dataset, tokenizer, batch_size, window_size=None, window_stride=None, drop_last=False):
    # process data: tokenize/slice/batch
    dataset = dataset.map(lambda s: tokenizer(s['text']), num_proc=4) # tokenize
    dataset_sliced = [slice_tokenized_datum(datum['input_ids'], window_size, window_stride) for datum in dataset] # slice
    dataset_sliced = list(itertools.chain.from_iterable(dataset_sliced)) # flatten
    dataset_sliced_batched, n_batch = batchify(dataset_sliced, batch_size, drop_last) # batch. this is an iterator
    return dataset_sliced_batched, n_batch


if __name__ == "__main__":
    # Example usage
    name = 'glue,ax,test'
    dataset = load_hf_dataset_with_sampling(name, n_samples=100)
    logger.info(f"Loaded {name} with {len(dataset)} samples.")