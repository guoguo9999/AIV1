﻿#  NVIDIA CORPORATION.

import os
import time
import copy
import json
import pickle
import random
import psutil
import PIL.Image
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter # Note: importing torchvision BEFORE tensorboard results in SIGSEGV
import torchvision
from src import dnnlib
from omegaconf import OmegaConf
from src.torch_utils import misc
from src.torch_utils import training_stats
from src.torch_utils.ops import conv2d_gradfix
from src.torch_utils.ops import grid_sample_gradfix

import src.legacy
from src.metrics import metric_main
from src.training.layers import sample_frames
from src.training.logging import generate_videos, save_video_frames_as_mp4

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(3840 // training_set.image_shape[2], 7, 32)
    gh = np.clip(2160 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]
    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    batches = [training_set[i] for i in grid_indices]
    images = [b['image'] for b in batches]
    labels = [b['label'] for b in batches]
    t = [b['times'] for b in batches]

    return (gw, gh), np.stack(images), np.stack(labels), np.stack(t)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname, quality=95)

#----------------------------------------------------------------------------

def training_loop(
    cfg                     = {},
    run_dir                 = '.',      # 输出文件目录
    training_set_kwargs     = {},       # 设置训练数据集的参数
    data_loader_kwargs      = {},       # 传递给torch.utils.data.DataLoader的参数
    G_kwargs                = {},       # 生成器网络的配置参数
    D_kwargs                = {},       # 判别器网络的参数
    G_opt_kwargs            = {},       # 生成器优化器的配置参数
    D_opt_kwargs            = {},       # 判别器优化器的配置参数
    augment_kwargs          = None,     # 数据增强管道的配置参数
    loss_kwargs             = {},       # 损失函数的配置参数
    metrics                 = [],       # 训练过程中需要评估的指标列表
    random_seed             = 0,        # 全局随机种子
    num_gpus                = 1,        # 参与训练的 GPU 数量
    rank                    = 0,        # 当前进程在所有参与训练的进程中的排名
    batch_size              = 4,        # 一次训练迭代中使用的总批量大小
    batch_gpu               = 4,        # 每个 GPU 一次处理的样本数量
    ema_kimg                = 10,       # 生成器权重的指数移动平均（EMA）的半衰期，以千张图像（kimg）为单位，默认为 10
    ema_rampup              = None,     # EMA 的渐进系数
    G_reg_interval          = None,        # 生成器正则化的执行频率
    D_reg_interval          = 16,       # 判别器正则化的执行频率
    augment_p               = 0,        # 数据增强的初始概率
    ada_target              = None,     # 自适应数据增强（ADA）的目标值
    ada_interval            = 4,        # 执行 ADA 调整的频率
    ada_kimg                = 500,      # ADA 调整的速度，以千张图像（kimg）为单位，表示概率 p 增加或减少一个单位所需的图像数量，默认为 500
    ada_linear              = False,    #是否使用线性的 ADA 调整方式
    total_kimg              = 25000,    # 训练的总长度，以千张真实图像（kimg）为单位，默认为 25000
    kimg_per_tick           = 5,        # 训练进度快照的间隔
    image_snapshot_ticks    = 50,       # 保存图像快照的频率，即每多少个 kimg_per_tick 保存一次图像快照
    network_snapshot_ticks  = 50,       # 保存网络模型快照的频率
    resume_pkl              = None,     # 用于恢复训练的网络模型 pickle 文件的路径
    resume_whole_state      = False,    # 是否恢复整个训练状态（包括优化器状态、训练进度等），还是仅恢复生成器、判别器和 EMA 生成器的模型参数
    cudnn_benchmark         = True,     # 是否启用 torch.backends.cudnn.benchmark，该选项可以提高训练速度
    allow_tf32              = False,    # 是否启用 torch.backends.cuda.matmul.allow_tf32 和 torch.backends.cudnn.allow_tf32，允许 PyTorch 在矩阵乘法和卷积操作中使用 TF32 数据类型
    abort_fn                = None,     # 用于判断是否中止训练的回调函数
    progress_fn             = None,     # 用于更新训练进度的回调函数，所有进程都会调用该函数
):
    # 初始化
    print('ada linear', ada_linear, flush=True)
    experiment_name = os.path.basename(os.path.dirname(run_dir))
    start_time = time.time()
    device = torch.device('cuda', rank)
    if random_seed is None:
       random_seed = int(time.time())  # 使用当前时间戳
       print(f"Using dynamic training seed: {random_seed}")
    random.seed(random_seed * num_gpus + rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # 提升训练速度
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # 允许 PyTorch 在内部将 tf32 用于矩阵乘法
    torch.backends.cudnn.allow_tf32 = allow_tf32        # 允许 PyTorch 内部使用 tf32 精度进行卷积运算
    conv2d_gradfix.enabled = True                       # 提升训练速度
    grid_sample_gradfix.enabled = True                  # 避免数据增强流程中的错误

    # 加载训练集
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num videos: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # 构建网络
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()
    # 从现有的pickle文件恢复训练
    if (resume_pkl is not None):
        if rank == 0:
            print(f'Resuming from "{resume_pkl}"')
        try:
            # 手动加载
            model_state_dcits = torch.load(resume_pkl, map_location=torch.device('cpu'))
            missing_keys, unexpected_keys = G.load_state_dict(model_state_dcits['G'], strict=False)
            G_ema = copy.deepcopy(G).eval()
            if rank == 0:
                print('============================================')
                print(f'missing keys when loading G: {missing_keys}')
                print(f'unexpected keys when loading G: {unexpected_keys}')
            b4_weight = model_state_dcits['D']['b4.out.weight']
            b4_bias = model_state_dcits['D']['b4.out.bias']
            model_state_dcits['D'].pop('b4.out.weight')
            model_state_dcits['D'].pop('b4.out.bias')
            missing_keys, unexpected_keys = D.load_state_dict(model_state_dcits['D'], strict=False)
            # 手动加载第4层到全连接层前的参数
            D.b4.out.weight.data.copy_(b4_weight)
            D.b4.out.bias.data.copy_(b4_bias)
            missing_keys.pop(missing_keys.index('b4.out.weight'))
            missing_keys.pop(missing_keys.index('b4.out.bias'))
            if rank == 0:
                print('============================================')
                print(f'missing keys when loading D: {missing_keys}')
                print(f'unexpected keys when loading D: {unexpected_keys}')
        except:
            with dnnlib.util.open_url(resume_pkl) as f:
                resume_data = src.legacy.load_network_pkl(f)
            if rank == 0:
                for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
                    misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
    else:
        resume_data = None
    #if (resume_pkl is not None):
    #    if rank == 0:
    #        print(f'Resuming from "{resume_pkl}"')
    #    with dnnlib.util.open_url(resume_pkl) as f:
    #        resume_data = src.legacy.load_network_pkl(f)

    #    if rank == 0:
    #        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
    #            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
    #else:
    #    resume_data = None

    cur_nimg = 0 if not resume_whole_state else resume_data['stats']['cur_nimg']
    cur_tick = 0 if not resume_whole_state else resume_data['stats']['cur_tick']
    batch_idx = 0 if not resume_whole_state else resume_data['stats']['batch_idx']
    tick_start_nimg = cur_nimg

    # 打印网络结构摘要表
    if rank == 0 and not resume_whole_state:
        z = torch.empty([2, G.z_dim], device=device) # [bf, z_dim]
        c = torch.empty([2, G.c_dim], device=device) # [b, c_dim]
        t = torch.zeros([2, cfg.sampling.num_frames_per_video], device=device).long() # [b, f]
        img = misc.print_module_summary(G, [z, c, t]) # [bf, c, h, w]
        misc.print_module_summary(D, [img, c, t])

    # 配置数据增强
    if rank == 0:
        print('Setting up augmentation...')

    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))

        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')
        else:
            ada_stats = None

        if resume_whole_state:
            misc.copy_params_and_buffers(resume_data['augment_pipe'], augment_pipe, require_all=False)
    else:
        augment_pipe = None
        ada_stats = None

    # 分布式训练
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    modules = [
        ('G_mapping', G.mapping),
        ('G_synthesis', G.synthesis),
        ('D', D),
        (None, G_ema),
        ('augment_pipe', augment_pipe),
    ]
    if cfg.model.loss_kwargs.motion_reg.coef > 0.0:
        modules.append(('G_motion_encoder', G.synthesis.motion_encoder))

    for name, module in modules:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module


    def get_params(module, is_G):
        if is_G:
            if hasattr(module, 'params_with_lr'):
                return module.params_with_lr(opt_kwargs.lr)
            ret = []
            ret += module.mapping.parameters()
            names = ['affine_w', 'motion_encoder', 'input', 'L0_36_512','L1_36_512', 'L2_36_512', 'L3_52_512', 'L4_52_512', 'L5_84_512', 'L6_84_512', 'L7_148_362', 'L8_148_256', 'L9_148_181',
                    'L10_276_128', 'L11_276_91', 'L12_276_64', 'L13_256_64', 'L14_256_3']
            for i in range(len(names) - cfg.model.generator.freezesyn):
                for param in getattr(module.synthesis, names[i]).parameters():
                    ret.append(param)
            return ret
        else:
            return module.params_with_lr(opt_kwargs.lr) if hasattr(module, 'params_with_lr') else module.parameters()

    # 配置训练阶段
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, **ddp_modules, **loss_kwargs) # training.loss.Loss的子类
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            params = module.params_with_lr(opt_kwargs.lr) if hasattr(module, 'params_with_lr') else module.parameters()
            # params = get_params(module, 'G' in name)
            opt = dnnlib.util.construct_class_by_name(params=params, **opt_kwargs) #torch.optim.Optimizer的子类
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # 延迟正则化
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            params = module.params_with_lr(opt_kwargs.lr) if hasattr(module, 'params_with_lr') else module.parameters()
            # params = get_params(module, 'G' in name)
            opt = dnnlib.util.construct_class_by_name(params=params, **opt_kwargs) # torch.optim.Optimizer的子类
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]

    for phase in phases:
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)
        else:
            phase.start_event = None
            phase.end_event = None
        phase.start_event_recorded = False
        phase.end_event_recorded = False

    G_opt = next(p.opt for p in phases if p.name in {'Gboth', 'Gmain', 'Greg'})
    D_opt = next(p.opt for p in phases if p.name in {'Dboth', 'Dmain', 'Dreg'})


    # 导出样本图像
    if rank == 0:
        if not resume_whole_state:
            vis = dnnlib.EasyDict(num_videos={128: 36, 256: 25, 512: 9, 1024: 1}[training_set.resolution])
            print('Exporting sample images...')
            vis.grid_size, images, vis.labels, vis.frames_idx = setup_snapshot_image_grid(training_set=training_set)
            save_image_grid(images[:, 0], os.path.join(run_dir, 'reals.jpg'), drange=[0,255], grid_size=vis.grid_size)
            vis.grid_z = torch.randn([vis.labels.shape[0], G.z_dim], device=device).split(batch_gpu) # (num_batches, [batch_size, z_dim])
            vis.grid_c = torch.from_numpy(vis.labels).to(device).split(batch_gpu) # (num_batches, [batch_size, c_dim])
            vis.grid_t = torch.from_numpy(vis.frames_idx).to(device).split(batch_gpu) # (num_batches, [batch_size, num_frames])
            images = torch.cat([G_ema(z=z, c=c, t=t[:, [0]], noise_mode='const').cpu() for z, c, t in zip(vis.grid_z, vis.grid_c, vis.grid_t)]).numpy()
            save_image_grid(images, os.path.join(run_dir, 'fakes_init.jpg'), drange=[-1,1], grid_size=vis.grid_size)

            # 生成视频训练数据
            assert len(vis.labels) >= vis.num_videos
            vis.video_len = {128: 150, 256: 150, 512: 32, 1024: 4}[training_set.resolution]
            vis.vgrid_z = torch.randn(vis.num_videos, G_ema.z_dim, device=device) # [batch_size, z_dim]
            vis.vgrid_z_addition= torch.randn(vis.num_videos, G_ema.z_dim, device=device) # [batch_size, z_dim]
            vis.vgrid_c = torch.from_numpy(vis.labels)[:vis.num_videos].to(device) # [batch_size, c_dim]
            vis.ts = torch.arange(vis.video_len, device=device).float().unsqueeze(0).repeat(vis.num_videos, 1) # [batch_size, video_len]
        else:
            vis = dnnlib.EasyDict(**resume_data['vis'])
            for k in vis:
                if isinstance(vis[k], torch.Tensor):
                    vis[k] = vis[k].to(device)
            images = torch.cat([G_ema(z=z, c=c, t=t[:, [0]], noise_mode='const').cpu() for z, c, t in zip(vis.grid_z, vis.grid_c, vis.grid_t)]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes_resume_{cur_nimg}.jpg'), drange=[-1,1], grid_size=vis.grid_size)
    else:
        vis = dnnlib.EasyDict()

    # 初始化日志
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            # stats_tfevents = SummaryWriter(run_dir)
            if not resume_whole_state:
                config_yaml = OmegaConf.to_yaml(cfg)
                # stats_tfevents.add_text(f'config', text_to_markdown(config_yaml), global_step=0, walltime=time.time())
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # 训练
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    if progress_fn is not None:
        progress_fn(cur_nimg, total_kimg)

    # 转为 bool 类型以优化 Hydra 的慢速访问问题
    use_fractional_t_for_G = bool(cfg.model.generator.motion.use_fractional_t)

    fvd16_best = 1e9
    fid_best = 1e9
    while True:
        # 获取训练数据
        with torch.autograd.profiler.record_function('data_fetch'):
            batch = next(training_set_iterator)
            phase_real_img, phase_real_c, phase_real_t, phase_real_l = batch['image'], batch['label'], batch['times'], batch['video_len']
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu) # [batch_gpu, batch_size, c_dim]
            phase_real_t = phase_real_t.to(device).split(batch_gpu) # [batch_gpu, batch_size, c_dim]
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]

            all_gen_z_addition = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z_addition = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z_addition.split(batch_size)]

            gen_cond_sample_idx = [np.random.randint(len(training_set)) for _ in range(len(phases) * batch_size)]
            all_gen_c = [training_set.get_label(i) for i in gen_cond_sample_idx]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]
            all_gen_l = [min(training_set.get_video_len(i), G.sampling_dict['max_num_frames']) for i in gen_cond_sample_idx]
            all_gen_t = [sample_frames(G.sampling_dict, use_fractional_t=use_fractional_t_for_G, total_video_len=l) for l in all_gen_l]
            all_gen_t = torch.from_numpy(np.stack(all_gen_t)).pin_memory().to(device)
            all_gen_t = [phase_gen_t.split(batch_gpu) for phase_gen_t in all_gen_t.split(batch_size)]

        # 启动分段训练
        for phase, phase_gen_z, phase_gen_z_addition, phase_gen_c, phase_gen_t in zip(phases, all_gen_z, all_gen_z_addition, all_gen_c, all_gen_t):
            if batch_idx % phase.interval != 0:
                continue

            # 梯度累积初始化
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
                phase.start_event_recorded = True
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)


            phase.module.train()

            # 多轮梯度累积
            curr_data = zip(phase_real_img, phase_real_c, phase_real_t, phase_gen_z, phase_gen_z_addition, phase_gen_c, phase_gen_t)
            for round_idx, (real_img, real_c, real_t, gen_z, gen_z_addition, gen_c, gen_t) in enumerate(curr_data):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval

                loss.accumulate_gradients(
                    phase=phase.name,
                    real_img=real_img,
                    real_c=real_c,
                    real_t=real_t,
                    gen_z=gen_z,
                    gen_c=gen_c,
                    gen_t=gen_t,
                    gen_z_addition=gen_z_addition,
                    sync=sync,
                    gain=gain)

            # 更新权重
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))
                phase.end_event_recorded = True

        # 滑动平均模型G_ema参数同步
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # 更新状态
        cur_nimg += batch_size * cfg.sampling.num_frames_per_video
        batch_idx += 1

        # 自适应数据增强
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            if ada_linear:
                new_prob = (ada_target / total_kimg) * (cur_nimg / 1e3)
                augment_pipe.p.copy_(misc.constant(new_prob, device=device))
            else:
                adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
                augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # 每刻执行一次维护任务
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # 打印状态行，并将相同信息累积到 stats_collector 中
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # 检查是否中止
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # 保存图像快照
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([G_ema(z=z, c=c, t=t[:, [0]], noise_mode='const').cpu() for z, c, t in zip(vis.grid_z, vis.grid_c, vis.grid_t)]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.jpg'), drange=[-1,1], grid_size=vis.grid_size)

            # 视频
            videos_diff_motion = generate_videos(G_ema, vis.vgrid_z, vis.vgrid_c, vis.ts, vis.vgrid_z_addition if cfg.training.interp_p > 0 else None, as_grids=True) # [video_len, 3, h, w]
            if not G_ema.synthesis.motion_encoder is None:
                with torch.no_grad():
                    motion_z = G_ema.synthesis.motion_encoder(c=vis.vgrid_c[[0]], t=vis.ts[[0]])['motion_z'] # [1, *motion_dims]
                    motion_z = motion_z.repeat_interleave(len(vis.ts), dim=0) # [batch_size, *motion_dims]
                    videos_same_motion = generate_videos(G_ema, vis.vgrid_z, vis.vgrid_c, vis.ts, vis.vgrid_z_addition if cfg.training.interp_p > 0 else None, motion_z=motion_z, as_grids=True) # [video_len, 3, h, w]

                assert videos_diff_motion.shape == videos_same_motion.shape, f"Wrong shape: {videos_diff_motion.shape} != {videos_same_motion.shape}"

                pad_size = 64
                videos_to_save = torch.cat([
                    videos_diff_motion,
                    torch.ones_like(videos_diff_motion[:, :, :, :pad_size]), # Some padding between the videos
                    videos_same_motion,
                ], dim=3) # [video_len, 3, h, w + pad_size + w]
            else:
                videos_to_save = videos_diff_motion

            videos_to_save = (videos_to_save * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, H, W, C]
            torchvision.io.write_video(os.path.join(run_dir, f'{experiment_name}_videos_{cur_nimg//1000:06d}.mp4'), videos_to_save, fps=cfg.dataset.fps, video_codec='h264', options={'crf': '10'})

        # 保存网络快照
        snapshot_pkl = None
        snapshot_data = None
        snapshot_modules = [
            ('G', G),
            ('D', D),
            ('G_ema', G_ema),
            ('augment_pipe', augment_pipe),
            ('G_opt', G_opt),
            ('D_opt', D_opt),
            ('vis', {k: (v.to('cpu') if isinstance(v, torch.Tensor) else v) for k, v in vis.items()}),
            ('stats', {'cur_nimg': cur_nimg, 'cur_tick': cur_tick, 'batch_idx': batch_idx}),
        ]
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            DDP_CONSISTENCY_IGNORE_REGEX = r'.*\.(w_avg|p|rnn\..*|embeds.*\.weight|num_batches_tracked|running_mean|running_var)'
            for name, module in snapshot_modules:
                if module is not None:
                    if isinstance(module, torch.nn.Module):
                        if 0 and num_gpus > 1:
                            misc.check_ddp_consistency(module, ignore_regex=DDP_CONSISTENCY_IGNORE_REGEX)
                            #misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                    else:
                        module = copy.deepcopy(module)
                snapshot_data[name] = module
                del module # 节省内存

        # 评估指标
        save_ckpt = False
        save_fid_ckpt = False
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print(f'Evaluating metrics for {experiment_name} ...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(
                    metric=metric,
                    G=snapshot_data['G_ema'],
                    dataset_kwargs=training_set_kwargs,
                    num_gpus=num_gpus,
                    rank=rank,
                    device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
                if metric == 'fvd2048_16f':
                    fvd16_value = result_dict['results']['fvd2048_16f']
                    if fvd16_value < fvd16_best:
                        fvd16_best = fvd16_value
                        save_ckpt = True
                if metric == 'fid50k_full':
                    fid_value = result_dict['results']['fid50k_full']
                    if fid_value < fid_best:
                        fid_best = fid_value
                        save_fid_ckpt = True


        if save_ckpt:
            snapshot_pkl = os.path.join(run_dir, f'best.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        if save_fid_ckpt:
            snapshot_pkl = os.path.join(run_dir, f'best_fid.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        del snapshot_data # 节省内存

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None) and phase.start_event_recorded and phase.end_event_recorded:
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # 更新日志
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()

        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # 更新状态
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------

def text_to_markdown(text: str) -> str:
    """
    Converts an arbitrarily text into a text that would be well-displayed in TensorBoard.
    TensorBoard uses markdown to render the text that's why it strips spaces and line breaks.
    This function fixes that.
    """
    text = text.replace(' ', '&nbsp;&nbsp;') # Because markdown does not support text indentation normally...
    text = text.replace('\n', '  \n') # Because tensorboard uses markdown

    return text

#----------------------------------------------------------------------------
