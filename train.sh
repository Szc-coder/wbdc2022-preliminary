python main.py --is_pretrain=True --max_epochs=5 --max_steps=78125 --warmup_steps=7812 --batch_size=64
python main.py --is_pretrain=True --max_epochs=4 --max_steps=62500 --warmup_steps=6250 --batch_size=64 --learning_rate=3e-5 --part2_pertrain=True
python kfTrain.py --savedmodel_path="../data/models/model5k/"
python kfTrain.py --model_name='albef' --savedmodel_path="../data/models/albef5k/"
python main.py --model_name='albef' --savedmodel_path="../data/models/albef_ema/" --double_attck=False
python main.py --model_name='albef' --savedmodel_path="../data/models/albef_ema_attck/" --double_attck=True --batch_size=16 --max_steps=25000 --warmup_steps=2500 --learning_rate=5e-5