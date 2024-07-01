import argparse
import os, sys
import pandas as pd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from lib.utils import DataLoaderX

from tensorboardX import SummaryWriter

import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import validate, run_fgsm_experiments, run_jsma_experiments, run_uap_experiments, run_ccp_experiments

from lib.core.general import fitness
from lib.models import get_net
from lib.utils.utils import create_logger, select_device


import datetime
import matplotlib.pyplot as plt

def plot_metrics(results_df, metric_list, x_param, attack_type, baseline_metrics):
    for metric in metric_list:
        plt.figure()
        plt.plot(results_df[x_param], results_df[metric], marker='o', label='Experiment Results')
        plt.axhline(y=baseline_metrics[metric], color='r', linestyle='--', label='Baseline')
        plt.xlabel(x_param)
        plt.ylabel(metric)
        plt.title(f'{metric} vs {x_param} for {attack_type}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{attack_type}_{metric}_plot.png')
        # plt.show()
        
def calculate_percentage_drop(initial, current):
    if initial == 0:
        return 0.0
    return ((initial - current) / initial) * 100

def parse_args():
    parser = argparse.ArgumentParser(description='Test Multitask network')

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='runs/')
    parser.add_argument('--weights',
                        nargs='+',
                        type=str,
                        default='/data2/zwt/wd/YOLOP/runs/BddDataset/detect_and_segbranch_whole/epoch-169.pth',
                        help ='model.pth path(s)')
    parser.add_argument('--conf_thres',
                        type=float,
                        default=0.001,
                        help ='object confidence threshold')
    parser.add_argument('--iou_thres',
                        type=float,
                        default=0.6,
                        help ='IOU threshold for NMS')
    
   # Adding new arguments for dataset and attack type
    parser.add_argument('--dataset',
                        type=str,
                        choices=['Carla', 'BDD100k'],
                        help ='Choice of dataset: Carla or BDD100k',
                        default='BDD100k')
    parser.add_argument('--attack',
                        type=str,
                        choices=['FGSM', 'JSMA', 'UAP', 'CCP', 'None'],
                        help ='Choice of attack: FGSM, JSMA, UAP, CCP, or None',
                        default= 'None')
    
    parser.add_argument('--experiment_mode',
                        type = int, choices = [0, 1],
                        help= 'Run with experiment mode? (1 (True): Runs with several pre-generated values. '
                          '0 (False): Provide your own parameters. FGSM: epsilon, attack type; '
                          'JSMA: num_pixels, perturbation value, attack type; '
                          'UAP: max_iterations, epsilon, delta, num_classes, targeted, batch_size; '
                          'CCP: epsilon, color_channel)',
                        default = 1)
    
    # New arguments for FGSM
    parser.add_argument('--epsilon',
                        type=float,
                        help='Epsilon value for FGSM or CCP attack',
                        default=0.1)
    parser.add_argument('--fgsm_attack_type', 
                        type=str, 
                        choices=['fgsm', 'fgsm_with_noise', 'iterative_fgsm'],
                        help ='Type of FGSM attack. Options include: FGSM, FGSM w Noise, and Iterative FGSM',
                        default='fgsm')
    
    # New arguments for JSMA
    parser.add_argument('--num_pixels',
                        type=int,
                        help="The number of pixels to be perturbed after saliency calculation.",
                        default = 10)
    parser.add_argument('--jsma_perturbation',
                        type=float,
                        help="The number of pixels to be perturbed after saliency calculation.",
                        default = .1)
    parser.add_argument('--jsma_attack_type',
                        type = str,
                        choices = ["Add", "Set", "Noise"],
                        help = "Select the type of perturbation to be applied to the highest scoring pixels. Options include add, set, and noise.",
                        default = "noise"
                        )
    
    # New arguments for UAP
    parser.add_argument('--uap_max_iterations',
                        type=int,
                        help='Maximum number of iterations for UAP attack',
                        default=10)
    parser.add_argument('--uap_eps',
                        type=float,
                        help='Epsilon value for UAP attack',
                        default=0.03)
    parser.add_argument('--uap_delta',
                        type=float,
                        help='Delta value for UAP attack',
                        default=0.8)
    parser.add_argument('--uap_num_classes',
                        type=int,
                        help='Number of classes for UAP attack',
                        default=None)
    parser.add_argument('--uap_targeted',
                        type=bool,
                        help='Whether the UAP attack is targeted or not',
                        default=False)
    parser.add_argument('--uap_batch_size',
                        type=int,
                        help='Batch size for UAP attack',
                        default=6)
    
    # New Args for CCP 
    parser.add_argument('--color_channel',
                    type=str,
                    choices=['R', 'G', 'B'],
                    help='Color channel to perturb (R, G, B)',
                    default='R')
    
    # Defenses 
    parser.add_argument('--resizer', type=str, help='Desired WIDTHxHEIGHT of your resized image')
    parser.add_argument('--quality', type=int, help='Desired quality for JPEG compression output. 0 - 100')
    parser.add_argument('--border_type', type=str, choices=['default', 'constant', 'reflect', 'replicate'], help= 'border type for Gaussian Blurring')
    parser.add_argument('--gauss', type=str, help="Apply Gaussian Blurring to image. Specify ksize as WIDTHxHEIGHT")
    parser.add_argument('--noise', type=float, help='Add Gaussian Noise to image. Specify sigma value for noise generation.')
    parser.add_argument('--bit_depth', type=int, help='Choose bit value between 1 - 8')

    args = parser.parse_args()
    return args

def create_and_save_table(results_df, normal_metrics, metrics, identifier_column, output_filename_prefix, identifier_values, display_identifier, combine=False):
    percentage_drops_df = results_df.copy()

    for metric in metrics:
        initial_value = normal_metrics[metric]
        percentage_drops_df[metric] = results_df[metric].apply(lambda x: calculate_percentage_drop(initial_value, x))

    if combine:
        display_df = results_df.copy()

        # Add normal metrics as the first row
        normal_metrics_row = normal_metrics.copy()
        normal_metrics_row[identifier_column] = display_identifier
        for metric in metrics:
            normal_metrics_row[f'{metric}_drop'] = 0
        
        normal_metrics_row_df = pd.DataFrame([normal_metrics_row])
        display_df = pd.concat([normal_metrics_row_df, display_df])

        # Round each metric to 4 significant figures
        for metric in metrics:
            display_df[metric] = display_df[metric].apply(lambda x: f'{x:.4g}')

        # Add percentage drops next to metrics
        for metric in metrics:
            display_df[f'{metric}_drop'] = percentage_drops_df[metric].apply(lambda x: f'{x:.2f}%')

        # Interleave the metric and drop columns
        interleaved_columns = []
        for metric in metrics:
            interleaved_columns.append(metric)
            interleaved_columns.append(f'{metric}_drop')

        interleaved_columns = [identifier_column] + interleaved_columns

        # Reorder the columns in display_df
        display_df = display_df[interleaved_columns]

        # Plotting the DataFrame
        fig, ax = plt.subplots(figsize=(28, 16))
        ax.axis('tight')
        ax.axis('off')

        # Create table
        table = ax.table(cellText=display_df.values, colLabels=display_df.columns, cellLoc='center', loc='center')

        # Style the drop columns to be red
        for (i, j), cell in table.get_celld().items():
            if j > 0 and display_df.columns[j].endswith('_drop'):
                cell.set_text_props(color='red')

        # Increase font size
        table.auto_set_font_size(False)
        table.set_fontsize(11)

        # Save the table as an image
        plt.savefig(f'{output_filename_prefix}_combined.png', bbox_inches='tight', dpi=600)
        plt.close(fig)

    else:
        for identifier_value in identifier_values:
            display_df = results_df[results_df[identifier_column] == identifier_value].copy()

            # Add normal metrics as the first row
            normal_metrics_row = normal_metrics.copy()
            normal_metrics_row[identifier_column] = display_identifier
            for metric in metrics:
                normal_metrics_row[f'{metric}_drop'] = 0
            
            normal_metrics_row_df = pd.DataFrame([normal_metrics_row])
            display_df = pd.concat([normal_metrics_row_df, display_df])

            # Round each metric to 4 significant figures
            for metric in metrics:
                display_df[metric] = display_df[metric].apply(lambda x: f'{x:.4g}')

            # Add percentage drops next to metrics
            for metric in metrics:
                display_df[f'{metric}_drop'] = percentage_drops_df[metric].apply(lambda x: f'{x:.2f}%')

            # Interleave the metric and drop columns
            interleaved_columns = []
            for metric in metrics:
                interleaved_columns.append(metric)
                interleaved_columns.append(f'{metric}_drop')

            interleaved_columns = [identifier_column] + interleaved_columns

            # Reorder the columns in display_df
            display_df = display_df[interleaved_columns]

            # Plotting the DataFrame
            fig, ax = plt.subplots(figsize=(28, 16))
            ax.axis('tight')
            ax.axis('off')

            # Create table
            table = ax.table(cellText=display_df.values, colLabels=display_df.columns, cellLoc='center', loc='center')

            # Style the drop columns to be red
            for (i, j), cell in table.get_celld().items():
                if j > 0 and display_df.columns[j].endswith('_drop'):
                    cell.set_text_props(color='red')

            # Increase font size
            table.auto_set_font_size(False)
            table.set_fontsize(11)

            # Save the table as an image
            plt.savefig(f'{output_filename_prefix}_{identifier_value}.png', bbox_inches='tight', dpi=600)
            plt.close(fig)
            
def main():
    # set all the configurations
    args = parse_args()
    update_config(cfg, args)
    
    # Attack type selection based on argument
    attack_type = args.attack
    
    if attack_type == 'None':
        attack_type = None
        print("None selected. Will run only a normal validation.")
    else:
        print(f"{attack_type} selected\n")
                

    # TODO: handle distributed training logger
    # set the logger, tb_log_dir means tensorboard logdir
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'test', attack_type=attack_type)
    
    print(logger)

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # bulid up model
    print("Begin to build up model...\n")
    
    # DP mode
    device = select_device(logger, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU* len(cfg.GPUS)) if not cfg.DEBUG \
        else select_device(logger, 'cpu')
    # device = select_device(logger, 'cpu')

    model = get_net(cfg)
    print("Finish build model\n")
    
    # define loss function (criterion) and optimizer
    criterion = get_loss(cfg, device=device)

    # load checkpoint model
    # det_idx_range = [str(i) for i in range(0,25)]
    model_dict = model.state_dict()
    checkpoint_file = args.weights[0] #args.weights
    logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file)
    checkpoint_dict = checkpoint['state_dict']
    
    # checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    logger.info("=> loaded checkpoint '{}' ".format(checkpoint_file))

    model = model.to(device)
    model.gr = 1.0
    model.nc = 1
    print("Build model finished")

    print("Begin to load data")
    # Data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    # Print available attributes in the dataset module
    # print(dir(dataset))
    
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers= 0, #cfg.WORKERS # Must be 0 or will cause pickling error
        pin_memory=False,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    
    print('Load data finished')
    
    epoch = 0 #special for test

    startTime = datetime.datetime.now()

    
    # Normal Validation    
    da_segment_results,ll_segment_results,detect_results, total_loss, maps, times = validate(
    epoch, cfg, valid_loader, valid_dataset, model, criterion,
    final_output_dir, tb_log_dir, writer_dict,
    logger, device
    )
    
    
    msg = 'Test:    Loss({loss:.3f})\n' \
      'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
      'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
      'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n' \
      'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
          loss=total_loss, da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],
          ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2],
          p=detect_results[0], r=detect_results[1], map50=detect_results[2], map=detect_results[3],
          t_inf=times[0], t_nms=times[1])
      
    logger.info(msg)
    
    normal_metrics = {
    'da_acc_seg': da_segment_results[0],
    'da_IoU_seg': da_segment_results[1],
    'da_mIoU_seg': da_segment_results[2],
    'll_acc_seg': ll_segment_results[0],
    'll_IoU_seg': ll_segment_results[1],
    'll_mIoU_seg': ll_segment_results[2],
    'detect_result': detect_results[2],  # mAP@0.5
    'loss_avg': total_loss,
    'da_acc_seg_drop': 0.0,
    'da_IoU_seg_drop': 0.0,
    'da_mIoU_seg_drop': 0.0,
    'll_acc_seg_drop': 0.0,
    'll_IoU_seg_drop': 0.0,
    'll_mIoU_seg_drop': 0.0,
    'detect_result_drop': 0.0,  # mAP@0.5
    'loss_avg_drop': 0.0
    }
    
    match attack_type:
        case "FGSM":
            # FGSM specific logic
            if args.experiment_mode == 1:
                epsilons = [.03, .05, .1, .15, .2, .3, .5, .75, .9, 1]  
                print(f"\nExperiment mode is {args.experiment_mode}, will be using pre-generated epsilon values of {epsilons}")
            elif args.experiment_mode == 0:
                epsilons = [args.epsilon]
                print(f"\nExperiment mode is {args.experiment_mode}, will be using your provided epsilon value of {epsilons}")

            fgsm_results_df = run_fgsm_experiments(
                model, valid_loader, device, cfg, criterion, epsilons, final_output_dir, args.fgsm_attack_type
            )
            
            metrics = ['da_acc_seg', 'da_IoU_seg', 'da_mIoU_seg', 'll_acc_seg', 'll_IoU_seg', 'll_mIoU_seg', 'loss_avg']

            create_and_save_table(fgsm_results_df, normal_metrics, metrics, 'epsilon', 'FGSM_results', fgsm_results_df['epsilon'].unique(), '0', combine=True)

            # Plotting the metrics
            plot_metrics(fgsm_results_df, metrics, 'epsilon', 'FGSM', normal_metrics)
        
        case "JSMA":
            # JSMA specific logic
            if args.experiment_mode:
                perturbation_params = [
                (10, 0.1, 'add'), (10, 0.1, 'set'), (10, 0.1, 'noise'),
                (10, 1.1, 'add'), (10, 1.1, 'set'), (10, 1.1, 'noise'),
                (60, 0.1, 'add'), (60, 0.1, 'set'), (60, 0.1, 'noise'),
                (60, 1.1, 'add'), (60, 1.1, 'set'), (60, 1.1, 'noise'),
                (110, 0.1, 'add'), (110, 0.1, 'set'), (110, 0.1, 'noise'),
                (110, 1.1, 'add'), (110, 1.1, 'set'), (110, 1.1, 'noise'),
                (160, 0.1, 'add'), (160, 0.1, 'set'), (160, 0.1, 'noise'),
                (160, 1.1, 'add'), (160, 1.1, 'set'), (160, 1.1, 'noise'),
                (160, 5.1, 'add'), (160, 5.1, 'set'), (160, 5.1, 'noise'),
                (160, 9.1, 'add'), (160, 9.1, 'set'), (160, 9.1, 'noise'),
                (610, 0.1, 'add'), (610, 0.1, 'set'), (610, 0.1, 'noise')
                ]

                print(len(perturbation_params))
                print(f"\nExperiment mode is on. Will run using pre-defined arguments of {perturbation_params}")
            else:
                perturbation_params = [(args.num_pixels, args.jsma_perturbation, args.jsma_attack_type)]
                print(f"\nExperiment mode is NOT on. Will run using provided arguments of {perturbation_params}")

            jsma_results_df = run_jsma_experiments(
                model, valid_loader, device, cfg, criterion, perturbation_params, final_output_dir            )
            metrics = ['da_acc_seg', 'da_IoU_seg', 'da_mIoU_seg', 'll_acc_seg', 'll_IoU_seg', 'll_mIoU_seg', 'loss_avg']
            create_and_save_table(jsma_results_df, normal_metrics, metrics, 'num_pixels', 'JSMA_results', jsma_results_df['num_pixels'].unique(), '0', combine= False)

            # Plotting the metrics
            plot_metrics(jsma_results_df, metrics, 'num_pixels', 'JSMA', normal_metrics)
        
        case "UAP":
            # UAP specific logic
            if args.experiment_mode:
                uap_params = [
                        (10, 0.1, 0.7, None, None, 10),
                        (10, 0.6, 0.75, None, None, 11),
                        (10, 1.1, 0.8, None, None, 12),
                        (10, 1.6, 0.85, None, None, 13),
                        (10, 2.1, 0.9, None, None, 14),
                        (10, 2.6, 0.95, None, None, 15),
                        (10, 3.1, 1.0, None, None, 16),
                        (10, 3.6, 1.05, None, None, 17),
                        (10, 4.1, 1.1, None, None, 18),
                        (10, 4.6, 1.15, None, None, 19),
                        (10, 5.1, 0.7, None, None, 20),
                        (10, 5.6, 0.75, None, None, 21),
                        (10, 6.1, 0.8, None, None, 22),
                        (10, 6.6, 0.85, None, None, 23),
                        (10, 7.1, 0.9, None, None, 24),
                        (10, 7.6, 0.95, None, None, 25),
                        (10, 8.1, 1.0, None, None, 26),
                        (10, 8.6, 1.05, None, None, 27),
                        (10, 9.1, 1.1, None, None, 28),
                        (10, 9.6, 1.15, None, None, 29)                   # Add more parameter sets as needed
                ]
                print(f"\nExperiment mode is on. Will run using pre-defined arguments of {uap_params}")
            else:
                uap_params = [(args.uap_max_iterations, args.uap_eps, args.uap_delta, args.uap_num_classes, args.uap_targeted, args.uap_batch_size)]
                print(f"\nExperiment mode is NOT on. Will run using provided arguments of {uap_params}")

            uap_results_df = run_uap_experiments(
                model, valid_loader, device, cfg, criterion, uap_params, final_output_dir
            )
            metrics = ['da_acc_seg', 'da_IoU_seg', 'da_mIoU_seg', 'll_acc_seg', 'll_IoU_seg', 'll_mIoU_seg', 'loss_avg']
            create_and_save_table(uap_results_df, normal_metrics, metrics, 'eps', 'UAP_results', uap_results_df['eps'].unique(), '0', combine= True)

            # Plotting the metrics
            plot_metrics(uap_results_df, metrics, 'eps', 'UAP', normal_metrics)
        
        case "CCP":
            # CCP specific logic
            if args.experiment_mode:
                ccp_params = [
                    # (0.01, 'R'),
                    # (0.03, 'R'),
                    # (0.05, 'R'),
                    # (0.1, 'R'),
                    # (0.5, 'R'),
                    # (1.0, 'R'),
                    # (2.5, 'R'),
                    # (5.0, 'R'),
                    # (10.0, 'R'),
                    # (15.0, 'R'),
                    # (0.01, 'G'),
                    # (0.03, 'G'),
                    # (0.05, 'G'),
                    # (0.1, 'G'),
                    # (0.5, 'G'),
                    # (1.0, 'G'),
                    # (2.5, 'G'),
                    # (5.0, 'G'),
                    # (10.0, 'G'),
                    # (15.0, 'G'),
                    (0.01, 'B'),
                    (0.03, 'B'),
                    (0.05, 'B'),
                    (0.1, 'B'),
                    (0.5, 'B'),
                    (1.0, 'B'),
                    (2.5, 'B'),
                    (5.0, 'B'),
                    (10.0, 'B'),
                    (15.0, 'B')
                    # Add more parameter sets as needed
                ]
                print(f"\nExperiment mode is on. Will run using pre-defined arguments of {ccp_params}")
            else:
                ccp_params = [(args.epsilon, args.color_channel)]
                print(f"\nExperiment mode is NOT on. Will run using provided arguments of {ccp_params}")

            ccp_results_df = run_ccp_experiments(
                model, valid_loader, device, cfg, criterion, ccp_params, final_output_dir)

            metrics = ['da_acc_seg', 'da_IoU_seg', 'da_mIoU_seg', 'll_acc_seg', 'll_IoU_seg', 'll_mIoU_seg', 'loss_avg']
            create_and_save_table(ccp_results_df, normal_metrics, metrics, 'epsilon', 'CCP_results', ccp_results_df['epsilon'].unique(), 'None', combine= True)

            # Plotting the metrics
            plot_metrics(ccp_results_df, metrics, 'epsilon', 'CCP', normal_metrics)



    print("Test Finish")
    
    print("Starting time: ")
    print(startTime)
    
    endTime = datetime.datetime.now()
    
    print("Ending time: ")
    print(endTime.strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == '__main__':
    main()
    
    

