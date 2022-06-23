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
        out_file = open('convert_labeled.json', 'w')
        json.dump(labeledData_convert, out_file)
        out_file.close()
    f.close()


convert_data('/root/autodl-tmp/data/annotations/')