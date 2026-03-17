"""
Search for governing equations of ant colony dynamics using ND2.

The script discovers symbolic expressions that describe how an ant's movement
is influenced by its own state and the aggregate influence of nearby ants.

Usage:
    # First convert the ANTS data:
    python scripts/convert_ants_data.py --gt_dir /path/to/Ant_dataset --output_dir ./data/ants

    # Then run the search:
    python scripts/search_ants.py --data Seq0001Object10Image94 --target vx --device cpu --time_limit 300
"""

import os
import json
import time
import signal
import logging
import warnings
import traceback
import numpy as np
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
from ND2.model import NDformer
from ND2.utils import init_logger, AutoGPU, seed_all
from ND2.search import MCTS
from ND2.GDExpr import GDExpr
from ND2.search.reward_solver import RewardSolver

warnings.filterwarnings("ignore", category=RuntimeWarning)
def handler(signum, frame): raise KeyboardInterrupt    
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)

logger = logging.getLogger('ND2.search')


class AntsRewardSolver(RewardSolver):
    def solve(self, prefix, *args, **kwargs):
        if prefix.count('aggr') > 1: return 0.0, None
        reward, coef_dict = super().solve(prefix, *args, **kwargs)
        return reward, coef_dict


def main(args):
    init_logger(args.name, f'./log/ants/{args.name}/info.log', root_name='ND2', info_level=args.info_level)
    setproctitle(f'{args.name}@ND2')
    if args.seed is None: args.seed = np.random.randint(0, 32768)
    seed_all(args.seed)
    if args.device == 'auto': args.device = AutoGPU().choice_gpu(900, interval=15, force=True)
    logger.info(f'Args: {args}')

    # %% Load Data
    data_path = f'./data/ants/{args.data}.json'
    raw = json.load(open(data_path, 'r'))
    
    px = np.array(raw['px'], dtype=np.float32)   # (T, V) normalized x-position
    py = np.array(raw['py'], dtype=np.float32)   # (T, V) normalized y-position
    A = np.array(raw['A'], dtype=int)             # (V, V) adjacency matrix
    G = np.array(raw['G'], dtype=int)             # (E, 2) edge list
    M = np.array(raw['M'], dtype=np.float32)      # (T, E) edge distance feature
    
    # Select target variable
    target = np.array(raw[args.target], dtype=np.float32)  # (T, V)
    
    # Build node and edge variable dicts
    Xv = {'px': px, 'py': py}
    Xe = {'M': M}
    vars_node = ['px', 'py']
    vars_edge = ['M']
    
    # Optionally add speed as node variable
    if 'speed' in raw and args.use_speed:
        speed = np.array(raw['speed'], dtype=np.float32)
        Xv['speed'] = speed
        vars_node.append('speed')
    
    logger.info(f'Data: {args.data}, Target: {args.target}')
    logger.info(f'Nodes (ants): {A.shape[0]}, Edges: {G.shape[0]}, Time steps: {px.shape[0]}')
    logger.info(f'Node vars: {vars_node}, Edge vars: {vars_edge}')

    # %% Init Model
    rewarder = AntsRewardSolver(
        Xv=Xv,
        Xe=Xe,
        A=A,
        G=G,
        Y=target,
        mask=None,
    )
    ndformer = NDformer(device=args.device)
    ndformer.load(args.model_path, weights_only=False)
    ndformer.eval()
    ndformer.set_data(
        Xv=Xv,
        Xe=Xe,
        A=A,
        G=G,
        Y=target,
        root_type='node',
        cache_data_emb=True
    )
    est = MCTS(
        rewarder=rewarder,
        ndformer=ndformer,
        vars_node=vars_node,
        vars_edge=vars_edge,
        binary=['add', 'sub', 'mul', 'div', 'regular'],
        unary=['neg', 'abs', 'inv', 'exp', 'logabs', 'sqrtabs', 'pow2', 'pow3', 'tanh', 'sigmoid', 'aggr', 'sour', 'targ'],
        log_per_episode=None,
        log_per_second=10,
        beam_size=10,
        use_random_simulate=False,
        max_token_num=30,
        max_coeff_num=5,
    )

    # %% Search
    try:
        logger.note(f'Start searching for {args.target} dynamics... Press ^C (Ctrl+C) to stop.')
        est.fit(
            ['node'],
            episode_limit=100_000_000, 
            time_limit=args.time_limit,
            early_stop=None,
        )
    except KeyboardInterrupt:
        logger.info(f'Interrupted manually.')
    except Exception:
        logger.error(traceback.format_exc())
    finally:
        log = {
            'Discovered': GDExpr.prefix2str(est.best_model),
            **est.best_metric,
        }
        logger.note(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in log.items()))
        pareto = est.Pareto()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default=f'AntsSearch_{time.strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('-d', '--device', type=str, default='auto')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('--data', type=str, default='Seq0001Object10Image94',
                        help='Sequence name (must have corresponding JSON in data/ants/)')
    parser.add_argument('--target', type=str, default='vx', choices=['vx', 'vy', 'speed'],
                        help='Target variable to predict')
    parser.add_argument('--use_speed', action='store_true',
                        help='Include speed as additional node variable')
    parser.add_argument('--model_path', type=str, default='./weights/checkpoint.pth')
    parser.add_argument('--time_limit', type=int, default=86400)
    parser.add_argument('--info_level', choices=['debug', 'info', 'note', 'warning', 'error', 'critical'], default='note')
    args, unknown = parser.parse_known_args()
    if unknown: warnings.warn(f'Unknown args: {unknown}')
    main(args)
