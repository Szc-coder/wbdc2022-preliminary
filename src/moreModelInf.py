import torch
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id, category_id_to_lv1id, category_id_to_lv2id
from model import MultiModal
from tqdm import tqdm
from orgModel import orgModel
from doubleModle import ALBEF

def inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 单流 5折
    model1 = MultiModal(args)
    checkpoint1 = torch.load('../data/models/model5k/k0_model_epoch_3.bin', map_location='cpu')
    model1.load_state_dict(checkpoint1['model_state_dict'])
    if torch.cuda.is_available():
        model1 = torch.nn.parallel.DataParallel(model1.cuda())
    model1.eval()

    model2 = MultiModal(args)
    checkpoint2 = torch.load('../data/models/model5k/k1_model_epoch_3.bin', map_location='cpu')
    model2.load_state_dict(checkpoint2['model_state_dict'])
    if torch.cuda.is_available():
        model2 = torch.nn.parallel.DataParallel(model2.cuda())
    model2.eval()
    
    model3 = MultiModal(args)
    checkpoint3 = torch.load('../data/models/model5k/k2_model_epoch_3.bin', map_location='cpu')
    model3.load_state_dict(checkpoint3['model_state_dict'])
    if torch.cuda.is_available():
        model3 = torch.nn.parallel.DataParallel(model3.cuda())
    model3.eval()
    
    model4 = MultiModal(args)
    checkpoint4 = torch.load('../data/models/model5k/k3_model_epoch_3.bin', map_location='cpu')
    model4.load_state_dict(checkpoint4['model_state_dict'])
    if torch.cuda.is_available():
        model4 = torch.nn.parallel.DataParallel(model4.cuda())
    model4.eval()

    model5 = MultiModal(args)
    checkpoint5 = torch.load('../data/models/model5k/k4_model_epoch_3.bin', map_location='cpu')
    model5.load_state_dict(checkpoint5['model_state_dict'])
    if torch.cuda.is_available():
        model5 = torch.nn.parallel.DataParallel(model5.cuda())
    model5.eval()

    # albef 5折
    model6 = ALBEF(args)
    checkpoint6 = torch.load('../data/models/albef5k/k0_model_epoch_2.bin', map_location='cpu')
    model6.load_state_dict(checkpoint6['model_state_dict'])
    if torch.cuda.is_available():
        model6 = torch.nn.parallel.DataParallel(model6.cuda())
    model6.eval()
    
    model7 = ALBEF(args)
    checkpoint7 = torch.load('../data/models/albef5k/k1_model_epoch_2.bin', map_location='cpu')
    model7.load_state_dict(checkpoint7['model_state_dict'])
    if torch.cuda.is_available():
        model7 = torch.nn.parallel.DataParallel(model7.cuda())
    model7.eval()
    
    model8 = ALBEF(args)
    checkpoint8 = torch.load('../data/models/albef5k/k2_model_epoch_2.bin', map_location='cpu')
    model8.load_state_dict(checkpoint8['model_state_dict'])
    if torch.cuda.is_available():
        model8 = torch.nn.parallel.DataParallel(model8.cuda())
    model8.eval()
    
    model9 = ALBEF(args)
    checkpoint9 = torch.load('../data/models/albef5k/k3_model_epoch_2.bin', map_location='cpu')
    model9.load_state_dict(checkpoint9['model_state_dict'])
    if torch.cuda.is_available():
        model9 = torch.nn.parallel.DataParallel(model9.cuda())
    model9.eval()
    
    model10 = ALBEF(args)
    checkpoint10 = torch.load('../data/models/albef5k/k4_model_epoch_2.bin', map_location='cpu')
    model10.load_state_dict(checkpoint10['model_state_dict'])
    if torch.cuda.is_available():
        model10 = torch.nn.parallel.DataParallel(model10.cuda())
    model10.eval()
    
    model11 = ALBEF(args)
    checkpoint11 = torch.load('../data/models/albef_ema_attck/model_epoch_3.bin', map_location='cpu')
    model11.load_state_dict(checkpoint11['model_state_dict'])
    if torch.cuda.is_available():
        model11 = torch.nn.parallel.DataParallel(model11.cuda())
    model11.eval()
    
    model12 = ALBEF(args)
    checkpoint12 = torch.load('../data/models/albef_ema/model_epoch_2.bin', map_location='cpu')
    model12.load_state_dict(checkpoint12['model_state_dict'])
    if torch.cuda.is_available():
        model12 = torch.nn.parallel.DataParallel(model12.cuda())
    model12.eval()

    
    # 3. inference
    predictions = []
    epoch_len = len(dataloader)
    with torch.no_grad():
        with tqdm(total=epoch_len) as t:
            for batch in dataloader:
                t.set_description('inference:')
                
                pred1= model1(batch, inference=True) * (1/6)
                pred2= model2(batch, inference=True) * (1/6)
                pred3= model3(batch, inference=True) * (1/6)
                pred4= model4(batch, inference=True) * (1/6)
                pred5= model5(batch, inference=True) * (1/6)
                pred6= model6(batch, inference=True) * (1/15)
                pred7= model7(batch, inference=True) * (1/15)
                pred8= model8(batch, inference=True) * (1/15)
                pred9= model9(batch, inference=True) * (1/15)
                pred10= model10(batch, inference=True) * (1/15)
                pred11= model11(batch, inference=True)  * (1/12)
                pred12= model12(batch, inference=True)  * (1/12)
                
                pred = pred1+pred2+pred3+pred4+pred5+pred6+pred7+pred8+pred9+pred10+pred11+pred12
                
                pred_label_id = torch.argmax(pred, dim=1)
                predictions.extend(pred_label_id.cpu().numpy())
                
                t.update(1)

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')
            
if __name__ == '__main__':
    inference()
