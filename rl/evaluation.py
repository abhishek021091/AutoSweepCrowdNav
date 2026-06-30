import numpy as np
import torch

from crowd_sim.envs.utils.info import *
from rl.pid import PID


def evaluate(actor_critic, eval_envs, num_processes, device, test_size, logging, config, args, visualize=False):
    """ function to run all testing episodes and log the testing metrics """
    # initializations
    eval_episode_rewards = []

    if config.robot.policy not in ['orca', 'social_force']:
        eval_recurrent_hidden_states = {}

        node_num = 1
        edge_num = actor_critic.base.human_num + 1
        eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, actor_critic.base.human_node_rnn_size,
                                                                     device=device)

        eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                           actor_critic.base.human_human_edge_rnn_size,
                                                                           device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    success_times = []
    collision_times = []
    timeout_times = []
    
    empty_arena = 0
    crowded_arena = 0
    success = 0
    collision = 0
    timeout = 0

    empty_episode_count = 0
    empty_success = 0
    empty_collision = 0
    empty_timeout = 0

    crowd_episode_count = 0
    crowd_success = 0
    crowd_collision = 0
    crowd_timeout = 0

    too_close_ratios = []
    min_dist = []

    collision_cases = []
    timeout_cases = []

    all_path_len = []

    # to make it work with the virtualenv in sim2real
    if hasattr(eval_envs.venv, 'envs'):
        baseEnv = eval_envs.venv.envs[0].env
    else:
        baseEnv = eval_envs.venv.unwrapped.envs[0].env
    time_limit = baseEnv.time_limit

    pid_x = PID(1.2, 0.0, 0.2)
    pid_y = PID(1.2, 0.0, 0.2)
    # start the testing episodes
    for k in range(test_size):
        baseEnv.episode_k = k
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0
        pid_x.reset()
        pid_y.reset()
        obs = eval_envs.reset()
        global_time = 0.0
        path_len = 0.
        too_close = 0.
        last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()


        while not done:
            stepCounter = stepCounter + 1
            if config.robot.controller == "ppo":

                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)

            elif config.robot.controller == "pid":

                robot = baseEnv.robot

                ex = robot.gx - robot.px
                ey = robot.gy - robot.py

                vx = pid_x.update(ex, baseEnv.time_step)
                vy = pid_y.update(ey, baseEnv.time_step)

                speed = np.hypot(vx, vy)

                if speed > robot.v_pref:
                    scale = robot.v_pref / speed
                    vx *= scale
                    vy *= scale

                action = torch.tensor(
                    [[vx, vy]],
                    dtype=torch.float32,
                    device=device
                )

            elif config.robot.policy in ['orca', 'social_force']:

                action = torch.zeros([1, 2], device=device)
            else:
                action = torch.zeros([1, 2], device=device)
            if not done:
                global_time = baseEnv.global_time

            # if the vec_pretext_normalize.py wrapper is used, send the predicted traj to env
            if args.env_name == 'CrowdSimPredRealGST-v0' and config.env.use_wrapper:
                out_pred = obs['spatial_edges'][:, :, 2:].to('cpu').numpy()
                # send manager action to all processes
                ack = eval_envs.talk2Env(out_pred)
                assert all(ack)
            # render
            if visualize:
                eval_envs.render()

            # Obser reward and next obs
            obs, rew, done, infos = eval_envs.step(action)

            # record the info for calculating testing metrics
            rewards.append(rew)

            path_len = path_len + np.linalg.norm(obs['robot_node'][0, 0, :2].cpu().numpy() - last_pos)
            last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()


            if isinstance(infos[0]['info'], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]['info'].min_dist)

            episode_rew += rew[0]


            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        # an episode ends!
        print('')
        print('Reward={}'.format(episode_rew))
        print('Episode', k, 'ends in', stepCounter)
        all_path_len.append(path_len)
        too_close_ratios.append(too_close/stepCounter*100)

        is_empty = infos[0].get('is_episode_empty_arena', getattr(baseEnv, 'current_empty_arena', False))
        if is_empty:
            empty_arena += 1
            empty_episode_count += 1
        else:
            crowded_arena += 1
            crowd_episode_count += 1

        if isinstance(infos[0]['info'], ReachGoal):
            success += 1
            if is_empty:
                empty_success += 1
            else:
                crowd_success += 1
            success_times.append(global_time)
            print('Success')
        elif isinstance(infos[0]['info'], Collision):
            collision += 1
            if is_empty:
                empty_collision += 1
            else:
                crowd_collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            print('Collision')
        elif isinstance(infos[0]['info'], Timeout):
            timeout += 1
            if is_empty:
                empty_timeout += 1
            else:
                crowd_timeout += 1
            timeout_cases.append(k)
            timeout_times.append(time_limit)
            print('Time out')
        elif infos[0]['info'] is None:
            pass
        else:
            raise ValueError('Invalid end signal from environment')

    # all episodes end
    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size
    assert success + collision + timeout == test_size
    avg_nav_time = sum(success_times) / len(
        success_times) if success_times else time_limit  # baseEnv.env.time_limit

    empty_success_rate = empty_success / empty_episode_count if empty_episode_count else np.nan
    empty_collision_rate = empty_collision / empty_episode_count if empty_episode_count else np.nan
    empty_timeout_rate = empty_timeout / empty_episode_count if empty_episode_count else np.nan

    crowd_success_rate = crowd_success / crowd_episode_count if crowd_episode_count else np.nan
    crowd_collision_rate = crowd_collision / crowd_episode_count if crowd_episode_count else np.nan
    crowd_timeout_rate = crowd_timeout / crowd_episode_count if crowd_episode_count else np.nan
    avg_min_dist = np.mean(min_dist) if min_dist else np.nan
    mean_eval_reward = np.mean(eval_episode_rewards) if eval_episode_rewards else np.nan

    # logging
    logging.info(
        'Is Empty Arena: {}, Is Crowded Arena: {}, Testing success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, '
        'nav time: {:.2f}, path length: {:.2f}, average intrusion ratio: {:.2f}%, '
        'average minimal distance during intrusions: {:.2f}'.
            format(empty_arena, crowded_arena, success_rate, collision_rate, timeout_rate, avg_nav_time, np.mean(all_path_len),
                   np.mean(too_close_ratios), avg_min_dist))
    logging.info(
        'Empty arena episodes: {}, success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}; '
        'Crowded arena episodes: {}, success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}'.
            format(empty_episode_count, empty_success_rate, empty_collision_rate, empty_timeout_rate,
                   crowd_episode_count, crowd_success_rate, crowd_collision_rate, crowd_timeout_rate))

    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), mean_eval_reward))

    eval_envs.close()
