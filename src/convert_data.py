import json
import os

def convert_data(ann_path):
    with open(os.path.join(ann_path, 'labeled.json'), 'r') as f:
        labeledData_convert = []
        data = json.load(f)
        for i in data:
            ocr = ''
            for k in i['ocr']:
                ocr += k['text']+'。'
            i['ocr'] = ocr
            labeledData_convert.append(i)
            
    with open(os.path.join(ann_path, 'convert_labeled.json'), 'w')as f2:
        json.dump(labeledData_convert, f2)

convert_data('data/annotations/')