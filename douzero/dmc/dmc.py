import os
import threading
import time
import timeit
import pprint
from collections import deque

import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .models import Model, OldModel
from .env_utils import get_device
from .utils import get_batch, log, create_env, create_buffers, create_optimizers, act

mean_episode_return_buf = {p:deque(maxlen=100) for p in ['landlord', 'landlord_up', 'landlord_down']}

def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets)**2).mean()
    return loss

def learn(position,
          actor_models,
          model,
          batch,
          optimizer,
          flags,
          lock):
    """Performs a learning (optimization) step."""
    device = get_device(flags.training_device)
    obs_x = batch['obs_x_no_action'].to(device)
    if 'obs_action' in batch:
        # old 编码：x = [状态特征, 动作特征]；resnet 编码的动作已在 obs_z 里，不再拼接
        obs_x = torch.cat((obs_x, batch['obs_action'].to(device)), dim=2)
    obs_x = torch.flatten(obs_x, 0, 1).float()
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    # 一个 unroll 窗口里可能没有任何 done=True（小 batch 尤其常见），
    # 对空张量取均值会得到 NaN，导致日志里的 mean_episode_return 永远是 nan
    if episode_returns.numel() > 0:
        mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))

    with lock:
        learner_outputs = model(obs_z, obs_x, return_value=True)
        loss = compute_loss(learner_outputs['values'], target)
        _return_buf = list(mean_episode_return_buf[position])
        stats = {
            'mean_episode_return_'+position: torch.mean(torch.stack(_return_buf)).item() if _return_buf else 0.0,
            'loss_'+position: loss.item(),
        }
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_models:
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats

def train(flags):  
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    # Initialize actor models
    models = []
    assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'
    # old 编码用 LSTM 系列，resnet/general 用 Model（ResnetModel）
    model_cls = OldModel if getattr(flags, 'model_type', 'resnet') == 'old' else Model
    for device in range(flags.num_actor_devices):
        model = model_cls(device=get_device(device))
        model.share_memory()
        model.eval()
        models.append(model)

    # Initialize buffers
    buffers = create_buffers(flags)
   
    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = []
    full_queue = []
    for device in range(flags.num_actor_devices):
        _free_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(), 'landlord_down': ctx.SimpleQueue()}
        _full_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(), 'landlord_down': ctx.SimpleQueue()}
        free_queue.append(_free_queue)
        full_queue.append(_full_queue)

    # Learner model for training
    learner_model = model_cls(device=get_device(flags.training_device))

    # Create optimizers
    optimizers = create_optimizers(flags, learner_model)

    # Stat Keys
    stat_keys = [
        'mean_episode_return_landlord',
        'loss_landlord',
        'mean_episode_return_landlord_up',
        'loss_landlord_up',
        'mean_episode_return_landlord_down',
        'loss_landlord_down',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'landlord':0, 'landlord_up':0, 'landlord_down':0}

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
                checkpointpath, map_location=get_device(flags.training_device)
        )
        for k in ['landlord', 'landlord_up', 'landlord_down']:
            learner_model.get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
            optimizers[k].load_state_dict(checkpoint_states["optimizer_state_dict"][k])
            for device in range(flags.num_actor_devices):
                models[device].get_model(k).load_state_dict(learner_model.get_model(k).state_dict())
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    # Starting actor processes
    for device in range(flags.num_actor_devices):
        num_actors = flags.num_actors
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device], models[device], buffers[device], flags))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device][position], full_queue[device][position], buffers[device][position], flags, local_lock)
            _stats = learn(position, models, learner_model.get_model(position), batch, 
                optimizers[position], flags, position_lock)

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                position_frames[position] += T * B

    for device in range(flags.num_actor_devices):
        for m in range(flags.num_buffers):
            free_queue[device]['landlord'].put(m)
            free_queue[device]['landlord_up'].put(m)
            free_queue[device]['landlord_down'].put(m)

    threads = []
    locks = [{'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()} for _ in range(flags.num_actor_devices)]
    position_locks = {'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()}

    for device in range(flags.num_actor_devices):
        for i in range(flags.num_threads):
            for position in ['landlord', 'landlord_up', 'landlord_down']:
                thread = threading.Thread(
                    target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,device,position,locks[device][position],position_locks[position]))
                thread.start()
                threads.append(thread)
    
    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        _models = learner_model.get_models()
        torch.save({
            'model_state_dict': {k: _models[k].state_dict() for k in _models},
            'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames
        }, checkpointpath)

        # Save the weights for evaluation purpose
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, position+'_weights_'+str(frames)+'.ckpt')))
            torch.save(learner_model.get_model(position).state_dict(), model_weights_dir)

    timer = timeit.default_timer
    interrupted = False
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint(frames)
                last_checkpoint_time = timer()

            end_time = timer()
            fps = (frames - start_frames) / (end_time - start_time)
            position_fps = {k:(position_frames[k]-position_start_frames[k])/(end_time-start_time) for k in position_frames}
            log.info('After %i (L:%i U:%i D:%i) frames: @ %.1f fps (L:%.1f U:%.1f D:%.1f) Stats:\n%s',
                         frames,
                         position_frames['landlord'],
                         position_frames['landlord_up'],
                         position_frames['landlord_down'],
                         fps,
                         position_fps['landlord'],
                         position_fps['landlord_up'],
                         position_fps['landlord_down'],
                         pprint.pformat(stats))

    except KeyboardInterrupt:
        interrupted = True
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)
    finally:
        # 旧代码从不结束 actor 进程：actor 里是 while True，主进程退出时会被
        # multiprocessing 的 atexit 逻辑 join 住，表现为「训练跑完了但命令行永远不返回」。
        for actor in actor_processes:
            actor.terminate()
        for actor in actor_processes:
            actor.join(timeout=5)

    if interrupted:
        return

    checkpoint(frames)
    plogger.close()
