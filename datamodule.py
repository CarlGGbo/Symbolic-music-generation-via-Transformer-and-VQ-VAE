import torch
import math
import os
from torch.utils.data import Dataset, DataLoader, IterableDataset
from util.vocab import RemiVocab
from torch.nn.utils.rnn import pad_sequence
from util.input_representation import InputRepresentation
#import pickle

def _get_split(files, worker_info):
  if worker_info:
    n_workers = worker_info.num_workers
    worker_id = worker_info.id

    per_worker = math.ceil(len(files) / n_workers)
    start_idx = per_worker*worker_id
    end_idx = start_idx + per_worker

    split = files[start_idx:end_idx]
  else:
    split = files
  return split


class Mydatamodule():
    def __init__(self,files,max_len,vae_module=None,batch_size=16,
               num_workers=2,**kwargs):
        self.a = 0
        self.train_val_test_split = (0.85, 0.1, 0.05)
        self.files = files
        self.max_len = max_len
        self.vae_module = vae_module
        self.pin_memory = True
        self.vocab = RemiVocab()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.kwargs = kwargs


    def setup(self, stage=None):

        n_valid = int(self.train_val_test_split[1] * len(self.files))
        n_test = int(self.train_val_test_split[2] * len(self.files))
        train_files = self.files[n_test + n_valid:]
        valid_files = self.files[n_test:n_test + n_valid]
        test_files = self.files[:n_test]

        self.train_ds = Mydataset(train_files, self.max_len,vae_module=self.vae_module,**self.kwargs)
        self.valid_ds = Mydataset(valid_files, self.max_len, vae_module=self.vae_module,**self.kwargs)
        self.test_ds = Mydataset(test_files, self.max_len, vae_module=self.vae_module,**self.kwargs)

        self.collator = SeqCollator(pad_token=self.vocab.to_i('<pad>'), context_size=self.max_len)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          collate_fn=self.collator,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_ds,
                          collate_fn=self.collator,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          collate_fn=self.collator,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers)

class SeqCollator:
    def __init__(self, pad_token=0, context_size=512):
        self.pad_token = pad_token
        self.context_size = context_size

    def __call__(self, features):
        batch = {}

        xs = [feature['input_ids'] for feature in features]

        xs = pad_sequence(xs, batch_first=True, padding_value=self.pad_token)

        if self.context_size > 0:
            max_len = self.context_size
            max_desc_len = self.context_size
        else:
            max_len = xs.size(1)
            max_desc_len = int(1e4)

        tmp = xs[:, :(max_len + 1)][:, :-1]
        labels = xs[:, :(max_len + 1)][:, 1:].clone().detach()
        xs = tmp

        seq_len = xs.size(1)

        batch['input_ids'] = xs
        batch['labels'] = labels



        if 'position_ids' in features[0]:
            position_ids = [feature['position_ids'] for feature in features]
            position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)
            batch['position_ids'] = position_ids[:, :seq_len]

        if 'bar_ids' in features[0]:
            bar_ids = [feature['bar_ids'] for feature in features]
            bar_ids = pad_sequence(bar_ids, batch_first=True, padding_value=0)
            batch['bar_ids'] = bar_ids[:, :seq_len]

        if 'file' in features[0]:
            batch['files'] = [feature['file'] for feature in features]
        if 'latents' in features[0]:
            latents = [feature['latents'] for feature in features]
            latents = pad_sequence(latents, batch_first=True, padding_value=0.0)
            batch['latents'] = latents[:, :max_desc_len]
    
        if 'codes' in features[0]:
            codes = [feature['codes'] for feature in features]
            codes = pad_sequence(codes, batch_first=True, padding_value=0)
            batch['codes'] = codes[:, :max_desc_len]
        return batch

class Mydataset(IterableDataset):
    def __init__(self,midi_files,max_len=512,vae_module=None,max_bars_per_context=-1,max_contexts_per_file=-1,
                 max_bars=512,description_flavor=['b'],max_positions=512,description_options=None):
        a = 1
        self.files = midi_files

        self.vocab = RemiVocab()
        self.description_flavor = description_flavor

        self.max_bars = max_bars
        self.print_errors = True
        self.max_positions = max_positions

        self.bar_token_mask = '<mask>'
        self.max_bars_per_context=max_bars_per_context
        self.max_len = max_len
        self.max_contexts_per_file=max_contexts_per_file

        self.latent_cache_path = None

        self.vae_module = vae_module


    def load_file(self, file):
        name = os.path.basename(file)

        rep = InputRepresentation(file, strict=True)
        events = rep.get_remi_events()
        description = rep.get_description()

        sample = {
            'events': events,
            'description': description
        }


        if self.description_flavor in ['latent', 'both']:
            latents, codes = self.get_latent_representation(sample['events'], name)
            sample['latents'] = latents
            sample['codes'] = codes


        return sample

    def get_bars(self, events, include_ids=False):
        bars = [i for i, event in enumerate(events) if f"Bar_" in event]

        if include_ids:
            bar_ids = torch.bincount(torch.tensor(bars, dtype=torch.int), minlength=len(events))
            bar_ids = torch.cumsum(bar_ids, dim=0)

            return bars, bar_ids
        else:
            return bars

    def get_positions(self, events):
        events = [f"Position_0" if f"Bar_" in event else event for event in events]
        position_events = [event if f"Position_" in event else None for event in events]

        positions = [int(pos.split('_')[-1]) if pos is not None else None for pos in position_events]

        if positions[0] is None:
            positions[0] = 0
        for i in range(1, len(positions)):
            if positions[i] is None:
                positions[i] = positions[i - 1]
        positions = torch.tensor(positions, dtype=torch.int)

        return positions

    def mask_bar_tokens(self, events, bar_token_mask='<mask>'):
        events = [bar_token_mask if f'Bar_' in token else token for token in events]
        return events

    def get_bos_eos_events(self, tuple_size=8):
        bos_event = torch.tensor(self.vocab.encode(['<bos>']), dtype=torch.long)
        eos_event = torch.tensor(self.vocab.encode(['<eos>']), dtype=torch.long)
        return bos_event, eos_event

    def get_latent_representation(self, events, cache_key=None, bar_token_mask='<mask>'):

        bars = self.get_bars(events)

        self.mask_bar_tokens(events, bar_token_mask=bar_token_mask)

        event_ids = torch.tensor(self.vocab.encode(events), dtype=torch.long)

        groups = [event_ids[start:end] for start, end in zip(bars[:-1], bars[1:])]
        groups.append(event_ids[bars[-1]:])

        bos, eos = self.get_bos_eos_events()

        self.vae_module.eval()

        for param in self.vae_module.parameters():
            param.requires_grad = False

        latents = []
        codes = []
        for bar in groups:
            x = torch.cat([bos, bar, eos])[:self.vae_module.context_size].unsqueeze(0)
            out = self.vae_module.encode(x)
            z, code = out['z'], out['codes']
            latents.append(z)
            codes.append(code)

        latents = torch.cat(latents)
        codes = torch.cat(codes)

        return latents.cpu(), codes.cpu()

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        self.split = _get_split(self.files, worker_info)

        split_len = len(self.split)

        for i in range(split_len):
            try:
                current_file = self.load_file(self.split[i])

            except ValueError as err:
                print('load file err')
                continue

            events = current_file['events']

            bars, bar_ids = self.get_bars(events, include_ids=True)

            if len(bars) > self.max_bars:
                if self.print_errors:
                    print(f"WARNING: REMI sequence has more than {self.max_bars} bars: {len(bars)} event bars.")
                continue

            # Identify positions
            position_ids = self.get_positions(events)
            max_pos = position_ids.max()
            if max_pos > self.max_positions:
                if self.print_errors:
                    print(
                        f"WARNING: REMI sequence has more than {self.max_positions} positions: {max_pos.item()} positions found")
                continue

            # Mask bar tokens if required
            if self.bar_token_mask is not None and self.max_bars_per_context > 0:
                events = self.mask_bar_tokens(events, bar_token_mask=self.bar_token_mask)

            # Encode tokens with appropriate vocabulary
            event_ids = torch.tensor(self.vocab.encode(events), dtype=torch.long)


            bos, eos = self.get_bos_eos_events()
            zero = torch.tensor([0], dtype=torch.int)

            if self.max_bars_per_context and self.max_bars_per_context > 0:
                # Find all indices where a new context starts based on number of bars per context
                starts = [bars[i] for i in range(0, len(bars), self.max_bars_per_context)]
                # Convert starts to ranges
                contexts = list(zip(starts[:-1], starts[1:])) + [(starts[-1], len(event_ids))]
                # # Limit the size of the range if it's larger than the max. context size
                # contexts = [(max(start, end - (self.max_len+1)), end) for (start, end) in contexts]

            else:
                event_ids = torch.cat([bos, event_ids, eos])
                bar_ids = torch.cat([zero, bar_ids, zero])
                position_ids = torch.cat([zero, position_ids, zero])


                if self.max_len > 0:
                    starts = list(range(0, len(event_ids), self.max_len + 1))
                    if len(starts) > 1:
                        contexts = [(start, start + self.max_len + 1) for start in starts[:-1]] + [
                            (len(event_ids) - (self.max_len + 1), len(event_ids))]
                    elif len(starts) > 0:
                        contexts = [(starts[0], self.max_len + 1)]
                else:
                    contexts = [(0, len(event_ids))]

            #print(contexts)   [len(256)(len(256))]

            if self.max_contexts_per_file and self.max_contexts_per_file > 0:
                contexts = contexts[:self.max_contexts_per_file]

            #print(event_ids[start:end])

            for start, end in contexts:
                # Add <bos> and <eos> to each context if contexts are limited to a certain number of bars
                if self.max_bars_per_context and self.max_bars_per_context > 0:
                    #print(event_ids[start:end])

                    src = torch.cat([bos, event_ids[start:end], eos])
                    b_ids = torch.cat([zero, bar_ids[start:end], zero])
                    p_ids = torch.cat([zero, position_ids[start:end], zero])

                else:
                    src = event_ids[start:end]
                    b_ids = bar_ids[start:end]
                    p_ids = position_ids[start:end]


                if self.max_len > 0:
                    src = src[:self.max_len + 1]

                x = {
                    'input_ids': src,
                    'file': os.path.basename(self.split[i]),
                    'bar_ids': b_ids,
                    'position_ids': p_ids,

                }
                if self.description_flavor in ['latent', 'both']:
                    x['latents'] = current_file['latents']
                    x['codes'] = current_file['codes']
                yield x