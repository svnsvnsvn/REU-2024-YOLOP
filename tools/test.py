import argparse
import os, sys
from glob import glob
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
from lib.core.function import AverageMeter, validate

# Attacks
from lib.core.Attacks.FGSM import validate_with_fgsm, run_fgsm_experiments
from lib.core.Attacks.JSMA import calculate_saliency, find_and_perturb_highest_scoring_pixels, validateJSMA, run_jsma_experiments

from lib.core.general import fitness
from lib.models import get_net
from lib.utils.utils import create_logger, select_device


from pathlib import Path
import json
import random
import cv2
from tqdm import tqdm
import math
import datetime
import plotly.graph_objects as go

from IPython.display import display, HTML

def calculate_percentage_drop(initial, current):
    return ((initial - current) / initial) * 100


# Flatten the dataframe by extracting relevant parts of the tuples/lists
def flatten_results(df):
    flat_data = []
    for _, row in df.iterrows():
        flat_row = {
            "num_pixels": row["num_pixels"],
            "perturb_value": row["perturb_value"],
            "attack_type": row["attack_type"],
            "loss_avg": row["loss_avg"],
            "time": row["time"][0],  # Assuming you want the first value from the time list
        }
        # Extracting values from tuples/lists
        flat_row["da_segment_iou"] = row["da_segment_result"][0]
        flat_row["da_segment_precision"] = row["da_segment_result"][1]
        flat_row["da_segment_recall"] = row["da_segment_result"][2]
        flat_row["ll_segment_iou"] = row["ll_segment_result"][0]
        flat_row["ll_segment_precision"] = row["ll_segment_result"][1]
        flat_row["ll_segment_recall"] = row["ll_segment_result"][2]
        flat_row["detect_accuracy"] = row["detect_result"][0]
        flat_row["detect_precision"] = row["detect_result"][1]
        flat_row["detect_recall"] = row["detect_result"][2]
        
        flat_data.append(flat_row)
    return pd.DataFrame(flat_data)

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
    parser.add_argument('--weights', nargs='+', type=str, default='/data2/zwt/wd/YOLOP/runs/BddDataset/detect_and_segbranch_whole/epoch-169.pth', help='model.pth path(s)')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    args = parser.parse_args()

    return args


    
def main():
    # set all the configurations
    args = parse_args()
    update_config(cfg, args)

    # TODO: handle distributed training logger
    # set the logger, tb_log_dir means tensorboard logdir
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'test')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # bulid up model
    # start_time = time.time()
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
        num_workers= 0, #cfg.WORKERS #This is the number of threads for data loading. Its 8 in config, I set it to 0 for a while due to a n error but now I will try 2 to see if that maybe speeds things up...Update: It did not speed anything up, it caused a pickling error
        pin_memory=False,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    
    print('Load data finished')
    
    # JSMA Attack Demonstration
    # saliency_scores = calculate_saliency(model, valid_loader, device, cfg, criterion)


    # Number of pixels to perturb
    # num_pixels_to_perturb = 0 #384 * 640  # Example value, adjust as needed
    # perturbation_value = 0.0 
    # perturbation_type = "add"
    
    # Find and perturb the highest scoring pixels
    '''perturbed_images, all_top_coords = find_and_perturb_highest_scoring_pixels(images, saliency_scores, num_pixels_to_perturb, perturbation_value, perturbation_type)
    print(f"\nPerturbed the top {num_pixels_to_perturb} pixels at coordinates for each image with a perturbation value of {perturbation_value}. Perturbation type: {perturbation_type}\n")'''
    
    startTime = datetime.datetime.now()

    
    epoch = 0 #special for test
    epsilons = [0, .03, .05, .1, .15, .2, .3, .5, .75, .9, 1]  # FGSM attack parameters
    
    #results_df = run_fgsm_experiments(model, valid_loader, device, cfg, criterion, epsilons)
    #results_df['epsilon'] = epsilons
    #initial_values = results_df[results_df['epsilon'] == 0]

    # percentage_drops = results_df.copy()
    # for metric in metrics:
    #     initial_value = initial_values[metric].values[0]
    #     percentage_drops[metric] = results_df[metric].apply(lambda x: calculate_percentage_drop(initial_value, x))

    # # Create a new DataFrame for displaying
    # display_df = pd.DataFrame({'epsilon': epsilons})
    # for metric in metrics:
    #     display_df[metric] = results_df[metric]
    #     display_df[f'{metric}_drop'] = percentage_drops[metric].apply(lambda x: f'<span style="color:red">{x:.2f}%</span>')

    # # Render the DataFrame as HTML
    # display(HTML(display_df.to_html(escape=False)))
    # # Creating a bar graph
    # fig = go.Figure()

    # # Add bar for each metric
    # metrics = ['da_acc_seg', 'da_IoU_seg', 'da_mIoU_seg', 'll_acc_seg', 'll_IoU_seg', 'll_mIoU_seg', 'detect_result', 'loss_avg', 'time']

    # for i, metric in enumerate(metrics):
    #     fig.add_trace(go.Bar(
    #         x=results_df['epsilon'],
    #         y=results_df[metric],
    #         name=metric,
    #         offsetgroup=i
    #     ))

    # # Update the layout
    # fig.update_layout(
    #     barmode='group',
    #     title='Impact of FGSM Epsilon on Model Metrics as Bar Graph',
    #     xaxis_title='Epsilon',
    #     yaxis_title='Metric Value',
    #     legend_title='Metrics',
    #     xaxis={'type': 'category'},  # This makes sure that epsilon values are treated as discrete categories
    #     yaxis={'range': [0, 1]}  # Assuming your metric values are normalized [0,1]
    # )

    # # Show the plot
    # fig.show()
    
    ''' # Normal Validation    
    da_segment_results,ll_segment_results,detect_results, total_loss, maps, times = validate(
    epoch,cfg, valid_loader, valid_dataset, model, criterion,
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
  
    
    # JSMA        
    da_segment_results,ll_segment_results,detect_results, total_loss, maps, times = validateJSMA(
                epoch,cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, perturbed_images, writer_dict,
                logger, device
    )'''
    
    perturbation_params = [
    (10, 0.1, 'add'),
    (10, 0.1, 'set'),
    (10, 0.1, 'noise')]
    (20, 0.1, 'add'),
    (20, 0.1, 'set'),
    (20, 0.1, 'noise'),
    (30, 0.1, 'add'),
    (30, 0.1, 'set'), 
    (30, 0.1, 'noise')
    # Add more combinations as needed
    # ]

    results_df = run_jsma_experiments(model, valid_loader, device, cfg, criterion, perturbation_params)
    metrics = ['da_acc_seg', 'da_IoU_seg', 'da_mIoU_seg', 'll_acc_seg', 'll_IoU_seg', 'll_mIoU_seg', 'loss_avg']
        
    # Calculate the percentage drop
    initial_values = results_df[results_df['num_pixels'] == 0]

    percentage_drops = results_df.copy()
    for metric in metrics:
        initial_value = initial_values[metric].values[0]
        percentage_drops[metric] = results_df[metric].apply(lambda x: calculate_percentage_drop(initial_value, x))

    # Create a new DataFrame for displaying
    display_df = pd.DataFrame({'num_pixels': results_df['num_pixels'], 'perturbation_type': results_df['perturbation_type']})
    for metric in metrics:
        display_df[metric] = results_df[metric]
        display_df[f'{metric}_drop'] = percentage_drops[metric].apply(lambda x: f'<span style="color:red">{x:.2f}%</span>')

    # Render the DataFrame as HTML
    display(HTML(display_df.to_html(escape=False)))

    '''
    # Creating a bar graph using Plotly
    figJSMA = go.Figure()

    # Define the new metrics
    metrics = ['da_acc_seg', 'da_IoU_seg', 'da_mIoU_seg', 'll_acc_seg', 'll_IoU_seg', 'll_mIoU_seg', 'loss_avg']
    
    # Add bar for each metric
    for metric in metrics:
        figJSMA.add_trace(go.Bar(
            x=results_df['num_pixels'],
            y=results_df[metric],
            name=metric
        ))

    # Update layout
    figJSMA.update_layout(
        title='Metrics vs Number of Pixels Perturbed',
        xaxis_title='Number of Pixels Perturbed',
        yaxis_title='Metric Value',
        xaxis={'type': 'category'},
        legend_title='Metrics',
        barmode='group'
    )

    # Show the figure
    figJSMA.show()'''
    # print(results_df.head())
    
    # import plotly.express as px
    
    # flattened_df = flatten_results(results_df)
    # print(flattened_df.head())  # Debug print

    # # Convert results to a format suitable for Plotly
    # results_long = flattened_df.melt(id_vars=["num_pixels", "perturb_value", "attack_type"], var_name="metric", value_name="value")
    # print(results_long.head())  # Debug print
    
    # unique_metrics = results_long['metric'].unique()
    
    # for metric in unique_metrics:
    #     metric_df = results_long[results_long['metric'] == metric]
        
    #     fig = px.line(
    #         metric_df,
    #         x="num_pixels",
    #         y="value",
    #         color="perturb_value",
    #         line_dash="attack_type",
    #         title=f"Effect of Perturbations on {metric}",
    #         labels={
    #             "num_pixels": "Number of Pixels",
    #             "value": metric,
    #             "perturb_value": "Perturbation Value"
    #         }
    #     )
        
    #     fig.show()

    # # Create an interactive plot
    # fig = px.line(
    #     results_long,
    #     x="perturb_value",
    #     y="value",
    #     color="metric",
    #     line_dash="attack_type",
    #     facet_col="num_pixels",
    #     title="JSMA Attack Performance with Different Parameters"
    # )
    
    # fig = px.bar(
    # results_long,
    # x="attack_type",  # This specifies the column to be used for the x-axis
    # y="value",        # This specifies the column to be used for the y-axis
    # color="metric",   # This specifies the column to be used for coloring the bars
    # facet_col="num_pixels",  # This creates subplots based on unique values in this column
    # title="Comparison of Attack Performance Across Different Types"
    # )

    # fig.show()
    # fig.update_xaxes(range=[0, 100])


    '''
    #fi = fitness(np.array(detect_results).reshape(1, -1))
    msg_jsma = 'JSMA Test:    Loss({loss:.3f})\n' \
            'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
            'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
            'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n' \
            'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                loss=total_loss, da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],
                ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2],
                p=detect_results[0], r=detect_results[1], map50=detect_results[2], map=detect_results[3],
                t_inf=times[0], t_nms=times[1])
    logger.info(msg_jsma)
    '''
    print("Test Finish")
    
    print("Starting time: ")
    print(startTime.strftime("%Y-%m-%d %H:%M:%S"))
    
    endTime = datetime.datetime.now()
    
    print("Ending time: ")
    print(endTime.strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == '__main__':
    main()
    
    

