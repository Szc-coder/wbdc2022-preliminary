import logging
import os
import time
import torch

from torch_ema import ExponentialMovingAverage
from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from tqdm import tqdm
from doubleModle import ALBEF

def validate(model, val_dataloader, epoch):
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


def train_and_validate(args):
    pertrain = args.is_pretrain
    
    # 1. load data
    if pertrain:
        train_dataloader = create_dataloaders(args, pertrain)
    else:
        train_dataloader, val_dataloader = create_dataloaders(args, pertrain)

    # 2. build model and optimizers
    if args.model_name == 'model':
        model = MultiModal(args, pertrain)
        if args.part2_pertrain:
            print('**********************loading part1 model**********************')
            checkpoint = torch.load(os.path.join(args.pertrain_mode_path_part1 ,'pretrain_epoch_4.bin'), map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = ALBEF(args)

    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
        
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    
    # 3. training
    step = 0
    epoch_len = len(train_dataloader)
    
    for epoch in range(args.max_epochs):
        with tqdm(total=epoch_len) as t:
            mean_loss, mean_acc = [], []
            for batch in train_dataloader:
                t.set_description('Epoch '+ str(epoch))
                model.train()
                if pertrain:
                    loss = model(batch)
                    loss = loss.mean()
                    mean_loss.append(loss)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    step += 1
                    t.set_postfix(mean_loss=(sum(mean_loss)/len(mean_loss)).item())
                    t.update(1)
                    if step % 100 == 0:
                        mean_loss.clear()
                else:
                    loss, accuracy, _, _ = model(batch)
                    loss = loss.mean()
                    accuracy = accuracy.mean()
                    loss.backward()
                    mean_loss.append(loss)
                    mean_acc.append(accuracy)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    step += 1
                    if step % args.print_steps == 0:
                        ema.update()
                    t.set_postfix(mean_loss=(sum(mean_loss)/len(mean_loss)).item(), mean_acc=(sum(mean_acc)/len(mean_acc)).item())
                    t.update(1)
                    if step % 100 == 0:
                        mean_loss.clear()
                        mean_acc.clear()

        if pertrain:
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            if args.part2_pertrain:
                torch.save({'epoch': epoch, 'model_state_dict': state_dict,},
                            f'{args.pertrain_mode_path_part2}/pretrain_epoch_{epoch}.bin')
            else:
                torch.save({'epoch': epoch, 'model_state_dict': state_dict,},
                            f'{args.pertrain_mode_path_part1}/pretrain_epoch_{epoch}.bin')
        else:          
            # 4. validation
            with ema.average_parameters():
                if args.is_alldata:
                    state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                    torch.save({'epoch': epoch, 'model_state_dict': state_dict,},
                                f'{args.savedmodel_path}/model_epoch_{epoch}.bin')
                else:
                    loss, results = validate(model, val_dataloader, epoch)
                    results = {k: round(v, 4) for k, v in results.items()}
                    logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

                    # 5. save checkpoint
                    mean_f1 = results['mean_f1']
                    best_score = mean_f1
                    state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                    torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                                f'{args.savedmodel_path}/model_epoch_{epoch}.bin')

def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    
    if os.path.exists(args.pertrain_mode_path_part1) == False:
        os.makedirs(args.pertrain_mode_path_part1)
        
    if os.path.exists(args.pertrain_mode_path_part2) == False:
        os.makedirs(args.pertrain_mode_path_part2)
    
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)    


if __name__ == '__main__':
    main()
