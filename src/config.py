import argparse
from ctypes.wintypes import BOOL


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')

    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='../data/annotations/labeled.json')
    parser.add_argument('--test_annotation', type=str, default='../data/annotations/test_b.json')
    parser.add_argument('--unlable_zip_feats', type=str, default='../data/annotations/unlabeled.json')
    parser.add_argument('--unlable_ann', type=str, default='../data/annotations/unlabeled.json')

    parser.add_argument('--train_zip_feats', type=str, default='../data/zip_feats/labeled.zip')
    parser.add_argument('--test_zip_feats', type=str, default='../data/zip_feats/test_b.zip')

    parser.add_argument('--test_output_csv', type=str, default='../result.csv')

    # ========================= 5F data Configs ==========================
    parser.add_argument('--convert_labeled_path', type=str, default='../data/convert_labeled.json')
    

    parser.add_argument('--val_ratio', default=0., type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=32, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=256, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=256, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='../data/models/model5k/')
    parser.add_argument('--ckpt_file', type=str, default='../autodl-tmp/save/double/model_epoch_2.bin')
    parser.add_argument('--best_score', default=0.5, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=4, help='How many epochs')
    parser.add_argument('--max_steps', default=11250, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1125, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=8e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='hfl/chinese-macbert-base')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=324)
    parser.add_argument('--bert_learning_rate', type=float, default=5e-5)
    parser.add_argument('--bert_warmup_steps', type=int, default=1125)
    parser.add_argument('--bert_max_steps', type=int, default=11250)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=8)
    parser.add_argument('--vlad_hidden_size', type=int, default=768, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')

    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=512, help="linear size before final linear")

    # ========================== update Code ==============================
    parser.add_argument('--is_pretrain', type=bool, default=False, help="is pertrain")
    parser.add_argument('--model_name', type=str, default="model")
    parser.add_argument('--pertrain_mode_path_part1', type=str, default="../data/models/pertrain_part1/")
    parser.add_argument('--pertrain_mode_path_part2', type=str, default="../data/models/pertrain_part2/")
    parser.add_argument('--part2_pertrain', type=bool, default=False)
    parser.add_argument('--double_attck', type=bool, default=False)

    return parser.parse_args()
