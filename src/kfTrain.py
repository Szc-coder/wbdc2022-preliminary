import random
import zipfile
import numpy as np
import torch
import pandas as pd
import os
import logging
import time

from config import parse_args
from pandas import read_json
from io import BytesIO
from functools import partial
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from category_id_map import category_id_to_lv2id
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from model import MultiModal
from doubleModle import ALBEF

def fold_train(total_df, args):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
    
    for k, (train_idx, val_idx) in enumerate(kf.split(total_df, total_df['category_id'])):
        print('**************flop:', k, '****************')
        train_df = total_df.iloc[train_idx, :]
        val_df = total_df.iloc[val_idx, :]

        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        kf_train(args, train_df, val_df, k)

        del train_df, val_df

        
        
class Kf_MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.
    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.
    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 data_frame,
                 zip_feats: str,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode

        self.zip_feat_path = zip_feats
        self.num_workers = args.num_workers
        if self.num_workers > 0:
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
        # load annotations
        self.anns = data_frame
        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns.loc[idx, 'id']
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_feats(idx)

        # Step 2, load title tokens
        ocr = self.anns.loc[idx, 'ocr']
        asr = self.anns.loc[idx, 'asr']
        title = self.anns.loc[idx, 'title']

        text = '[CLS]' + title + '[SEP]' + asr + '[SEP]' + ocr + '[SEP]'

        title_input, title_mask = self.tokenize_text(text)

        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            text_input=title_input,
            text_mask=title_mask
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns.loc[idx, 'category_id'])
            data['label'] = torch.LongTensor([label])

        return data

    
def create_k_dataloaders(args, train_df, val_df):
    train_dataset = Kf_MultiModalDataset(args, train_df, args.train_zip_feats)
    val_dataset = Kf_MultiModalDataset(args, val_df, args.train_zip_feats)

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)

    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)

    return train_dataloader, val_dataloader

def k_validate(model, val_dataloader, epoch):
    model.eval()
    predictions = []
    labels = []
    losses = []
    epoch_len = len(val_dataloader)
    with torch.no_grad():
        with tqdm(total=epoch_len) as t:
            for batch in val_dataloader:
                t.set_description('Epoch '+ str(epoch))
                loss, accuracy, pred_label_id, label = model(batch)
                loss = loss.mean()
                predictions.extend(pred_label_id.cpu().numpy())
                labels.extend(label.cpu().numpy())
                losses.append(loss.cpu().numpy())
                t.set_postfix(loss=loss.item(), accuracy=accuracy.item())
                t.update(1)
                
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def kf_train(args, train_df, val_df, k):

    train_dataloader, val_dataloader = create_k_dataloaders(args, train_df, val_df)

    if args.model_name == 'model':
        print('**********************loading pretrain model**********************')
        model = MultiModal(args)
        checkpoint = torch.load(os.path.join(args.pertrain_mode_path_part2 ,'pretrain_epoch_3.bin'), map_location='cpu')
        model.load_state_dict({'module.'+k:v for k,v in checkpoint['model_state_dict'].items()}, False)
    else:
        model = ALBEF(args)
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
        
    # 3. training
    step = 0
    epoch_len = len(train_dataloader)
    
    for epoch in range(args.max_epochs):
        with tqdm(total=epoch_len) as t:
            
            mean_loss, mean_acc = [], []
            model.train()
            
            for batch in train_dataloader:
                t.set_description('Epoch '+ str(epoch))
                loss, accuracy, _, _ = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                loss.backward()
                
                mean_loss.append(loss)
                mean_acc.append(accuracy)
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                t.set_postfix(mean_loss=(sum(mean_loss)/len(mean_loss)).item(), mean_acc=(sum(mean_acc)/len(mean_acc)).item())
                t.update(1)
                
                step += 1
                if step % 100 == 0:
                    mean_loss.clear()
                    mean_acc.clear()
       

            loss, results = k_validate(model, val_dataloader, epoch)
            results = {k: round(v, 4) for k, v in results.items()}
            logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

            mean_f1 = results['mean_f1']
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                        f'{args.savedmodel_path}/k{k}_model_epoch_{epoch}.bin')

            
            
def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)
    
    total_df = pd.DataFrame(read_json(args.convert_labeled_path, dtype={'id':str, 'category_id':str}))
    
    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    fold_train(total_df, args)    


if __name__ == '__main__':
    main()
