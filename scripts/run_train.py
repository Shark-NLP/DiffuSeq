import sys
import os
import argparse
import time
sys.path.append('.')

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--dataset', type=str, default='', help='name of training dataset')
    parser.add_argument('--data_dir', type=str, default='', help='path to training dataset')
    parser.add_argument('--data_split_num', type=int, default=0, help='fold number of training dataset')

    parser.add_argument('--noise_schedule', type=str, default='cosine', choices=['linear', 'cosine', 'sqrt', 'trunc_cos', 'trunc_lin', 'pw_lin', 'warmup-decay'], help='the distribution of noises')
    parser.add_argument('--diff_steps', type=int, default=4000, help='diffusion steps')
    parser.add_argument('--schedule_sampler', type=str, default='uniform', choices=['uniform', 'lossaware', 'fixstep'], help='schedule sampler of timesteps')
    parser.add_argument('--learned_mean_embed', default=False, type=str2bool, help='learn mean embed for gaussian; default is zero-mean gaussian')
    parser.add_argument('--denoise', default=False, type=str2bool, help='denoise combined with learned_mean_embed [MASK]')
    parser.add_argument('--reg_rate', default=0.0, type=float, help='regularization rate of learned mean embed for gaussian; default is zero')
    parser.add_argument('--denoise_rate', default=0.2, type=float, help='max denoise rate of [MASK]')

    parser.add_argument('--seq_len', type=int, default=128, help='max len of input sequence')
    parser.add_argument('--hidden_t_dim', type=int, default=128, help='hidden size of time embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden size of word embedding')
    parser.add_argument('--learning_steps', type=int, default=40000, help='total steps of learning')
    parser.add_argument('--save_interval', type=int, default=10000, help='save step')
    parser.add_argument('--resume_checkpoint', type=str, default='none', help='path to resume checkpoint, like xxx/xxx.pt')
    parser.add_argument('--lr', type=float, default=1e-04, help='learning rate')
    parser.add_argument('--use_fp16', action='store_true', help='use fp16 or not')
    parser.add_argument('--bsz', type=int, default=64, help='batch size')
    parser.add_argument('--microbatch', type=int, default=64, help='microbatch size')
    parser.add_argument('--seed', type=int, default=101, help='random seed')

    parser.add_argument('--config_name', type=str, default='bert-base-uncased', help='config of pre-trained models')
    parser.add_argument('--vocab', type=str, default='bert', help='use bert vocab or load external vocab dict if given as path')
    parser.add_argument('--use_plm_init', type=str, default='no', choices=['no', 'bert'], help='load init parameter from the pre-trained lm')

    parser.add_argument('--notes', type=str, default='-', help='as training notes or specifical args')
    parser.add_argument('--app', type=str, default='', help='other input args')
    
    args = parser.parse_args()

    # set working dir to the upper folder
    abspath = os.path.abspath(sys.argv[0])
    dname = os.path.dirname(abspath)
    dname = os.path.dirname(dname)
    os.chdir(dname)

    folder_name = "diffusion_models/"

    if int(os.environ['LOCAL_RANK']) == 0:
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

    Model_FILE = f"diffuseq_{args.dataset}_h{args.hidden_dim}_lr{args.lr}" \
                f"_t{args.diff_steps}_{args.noise_schedule}_{args.schedule_sampler}" \
                f"_seed{args.seed}"
    if args.notes:
        args.notes += time.strftime("%Y%m%d-%H:%M:%S")
        Model_FILE = Model_FILE + f'_{args.notes}'
    Model_FILE = os.path.join(folder_name, Model_FILE)

    if int(os.environ['LOCAL_RANK']) == 0:
        if not os.path.isdir(Model_FILE):
            os.mkdir(Model_FILE)

    COMMANDLINE = f" OPENAI_LOGDIR={Model_FILE}  " \
                  f"TOKENIZERS_PARALLELISM=false " \
                  f"python train.py   " \
                  f"--checkpoint_path {Model_FILE} " \
                  f"--dataset {args.dataset} --data_dir {args.data_dir} --data_split_num {args.data_split_num} --vocab {args.vocab} --use_plm_init {args.use_plm_init} " \
                  f"--lr {args.lr} --use_fp16 {args.use_fp16} " \
                  f"--batch_size {args.bsz} --microbatch {args.microbatch} " \
                  f"--diffusion_steps {args.diff_steps} " \
                  f"--noise_schedule {args.noise_schedule} " \
                  f"--schedule_sampler {args.schedule_sampler} --resume_checkpoint {args.resume_checkpoint} " \
                  f"--seq_len {args.seq_len} --hidden_t_dim {args.hidden_t_dim} --seed {args.seed} " \
                  f"--hidden_dim {args.hidden_dim} " \
                  f"--learning_steps {args.learning_steps} --save_interval {args.save_interval} " \
                  f"--config_name {args.config_name} --notes {args.notes} " \
                  f"--learned_mean_embed {args.learned_mean_embed} " \
                  f"--denoise {args.denoise} --denoise_rate {args.denoise_rate} " \
                  f"--reg_rate {args.reg_rate} "

    COMMANDLINE += " " + args.app

    if int(os.environ['LOCAL_RANK']) == 0:
        with open(os.path.join(Model_FILE, 'saved_bash.sh'), 'w') as f:
            print(COMMANDLINE, file=f)

    print(COMMANDLINE)
    os.system(COMMANDLINE)