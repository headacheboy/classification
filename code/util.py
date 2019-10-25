from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='pytorch/mtl')
    parser.add_argument('--model_name', type=str, default="CNN")
    parser.add_argument('--add_no_rev_grad', type=bool, default=False)
    parser.add_argument('--lr_decay', type=int, default=4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.8)
    parser.add_argument('--lr_decay_begin', type=int, default=0)
    parser.add_argument('--decay_method', type=str, default="linear")
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--out_channel', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--text_emb', type=int, default=200)
    parser.add_argument('--label_num', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout_keep_rate', type=float, default=0.5)
    parser.add_argument('--display_step', type=int, default=175)
    parser.add_argument('--valid_step', type=int, default=175)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--seqlen', type=int, default=5940)
    parser.add_argument('--lstm_layer_size', type=int, default=1)
    parser.add_argument('--use_unlabel', type=bool, default=False)
    parser.add_argument('--task', type=int, default=16)
    parser.add_argument('--lamb', type=float, default=0.1)
    parser.add_argument('--maxClip', type=float, default=5.0)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--multi_att_len', type=int, default=2)
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--decay', type=float, default=0.9)
    parser.add_argument('--reatt', type=int, default=0)
    parser.add_argument('--wordemb_suffix', type=str, default="t")
    parser.add_argument('--mask_prob', type=float, default=0.15)
    parser.add_argument('--cross', type=int, default=None)
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--pretrain_path', type=str, default=None)
    args = parser.parse_args()
    args.task_len_dict={
            "apparel": 1202, 
            "baby": 847,
            "books": 5952, 
            "camera_photo": 2623,
            "dvd": 5316,
            "electronics": 2018,
            "health_personal_care": 1193, 
            "imdb": 1187, 
            "kitchen_housewares": 1383, 
            "magazines": 2641, 
            "MR": 61, 
            "music": 1668, 
            "software": 4791, 
            "sports_outdoors": 1218, 
            "toys_games": 1123, 
            "video": 1551
    }
    #args.task_len=[1202, 847, 5940, 2623, 5316, 2018, 1635, 1951, 1382, 2640, 54, 1668, 4791, 1356, 1123, 2185]
    args.task_len = [512] * 16
    #args.task_len = [0 for i in range(args.task)]
    return args

