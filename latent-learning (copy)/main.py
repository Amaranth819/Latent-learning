import argparse
import os
import yaml
from ppo import PPO
# from dynamics.halfcheetah import HalfCheetahEnv_ChangeDynamics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    '''
        For PPO hyperparameters
    '''
    ppo_parser = parser.add_argument_group('ppo')

    # Env
    ppo_parser.add_argument('--env_id', type = str, default = 'HalfCheetah-v4')
    ppo_parser.add_argument('--n_envs', type = int, default = 8)
    ppo_parser.add_argument('--env_class', type = str, default = None)

    # Actor-Critic
    ppo_parser.add_argument('--actor_lr', type = float, default = 0.0005)
    ppo_parser.add_argument('--critic_lr', type = float, default = 0.0002)
    ppo_parser.add_argument('--worldmodel_lr', type = float, default = 0.0003)
    ppo_parser.add_argument('--worldmodel_class', type = str, default = 'dSSMVAE')
    ppo_parser.add_argument('--is_worldmodel_deterministic', action='store_true')
    ppo_parser.add_argument('--state_dim', type = int, default = 128)
    ppo_parser.add_argument('--hidden_dim', type = int, default = 128)
    ppo_parser.add_argument('--num_gru_layers', type = int, default = 2)
    ppo_parser.add_argument('--fixed_action_std', type = float, default = 0.5)
    # ppo_parser.add_argument('--min_state_std', type = float, default = 0.1)
    # ppo_parser.add_argument('--max_state_std', type = float, default = 1.0)

    # Hyperparameters
    ppo_parser.add_argument('--batch_size', type = int, default = 32)
    ppo_parser.add_argument('--seq_len', type = int, default = 50)
    ppo_parser.add_argument('--collect_intervals', type = int, default = 20)
    ppo_parser.add_argument('--repeat_batch', type = int, default = 2)
    ppo_parser.add_argument('--epsilon', type = float, default = 0.2)
    ppo_parser.add_argument('--gamma', type = float, default = 0.99)
    ppo_parser.add_argument('--gae_lambda', type = float, default = 0.95)
    ppo_parser.add_argument('--vloss_coef', type = float, default = 0.1)
    ppo_parser.add_argument('--recon_obs_loss_coef', type = float, default = 0.1)
    ppo_parser.add_argument('--clip_loss_coef', type = float, default = 100.0)
    ppo_parser.add_argument('--actor_entropy_coef', type = float, default = 10.0)
    ppo_parser.add_argument('--device', type = str, default = 'auto')

    '''
        For training setup
    '''
    training_parser = parser.add_argument_group('training')

    # Training settings
    training_parser.add_argument('--load_from_path', type = str, default = None)
    training_parser.add_argument('--save_root_dir', type = str, default = './WorldModelPPO/')
    training_parser.add_argument('--epochs', type = int, default = 500)
    training_parser.add_argument('--eval_frequency', type = int, default = None)

    '''
        For recording videos
    '''
    recording_parser = parser.add_argument_group('record')
    recording_parser.add_argument('--load_model', type = str, default = None)
    recording_parser.add_argument('--target_video_path', type = str, default = None)

    '''
        Parse the arguments
    '''
    config = parser.parse_args()

    for group in parser._action_groups:
        if group.title == 'ppo':
            ppo_dict = {a.dest : getattr(config, a.dest, a.default) for a in group._group_actions}
            print('ppo', ppo_dict)
        elif group.title == 'training':
            training_dict = {a.dest : getattr(config, a.dest, a.default) for a in group._group_actions}
            print('training', training_dict)
        elif group.title == 'record':
            record_dict = {a.dest : getattr(config, a.dest, a.default) for a in group._group_actions}
            print('record', record_dict)

    # Save the configuration file to .yaml format
    if not os.path.exists(training_dict['save_root_dir']):
        os.mkdir(training_dict['save_root_dir'])

    with open(os.path.join(training_dict['save_root_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Create PPO
    if ppo_dict['env_class'] != None:
        ppo_dict['env_class'] = eval()
    algo = PPO(**ppo_dict)

    if training_dict['load_from_path']:
        algo.load(training_dict['load_from_path'])

    # Learning
    log_path = os.path.join(training_dict['save_root_dir'], 'log/')
    best_model_path = os.path.join(training_dict['save_root_dir'], 'PPO_best.pkl')
    algo.learn(
        epochs = training_dict['epochs'],
        eval_frequency = training_dict['eval_frequency'],
        log_path = log_path,
        best_model_path = best_model_path
    )

    # Save
    target_model_path = os.path.join(training_dict['save_root_dir'], 'PPO.pkl')
    algo.save(target_model_path)

    # Record video
    if record_dict['target_video_path'] is not None:
        print('Recording a video!')
        if record_dict['load_model'] is not None:
            algo.load(record_dict['load_model'])
        algo.record_video(record_dict['target_video_path'])
        print('Eval:', algo.collect(True))