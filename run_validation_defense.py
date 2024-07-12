import argparse
import os
import json
import pprint
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from lib.utils import DataLoaderX
from tensorboardX import SummaryWriter
import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import validate
from lib.models import get_net
from lib.utils.utils import create_logger, select_device
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Run validation for different attacks and defenses")
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
                        default='weights/End-to-end.pth',
                        help='model.pth path(s)')
    parser.add_argument('--defended_images_dir',
                        help='path to the Defended Images directory',
                        type=str,
                        default='DefendedImages')
    return parser.parse_args()

def run_validation(cfg, args, attack_params, defense_params):
    attack_type = attack_params['attack_type']
    defense_type = defense_params

    cfg.defrost()
    cfg.DATASET.TEST_SET = f'{attack_type}/{defense_type}'
    cfg.freeze()

    # set the logger, tb_log_dir means tensorboard logdir
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'test', attack_type=attack_type, defense_type=defense_type)

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # Build the model
    device = select_device(logger, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)) if not cfg.DEBUG else select_device(logger, 'cpu')
    model = get_net(cfg)

    # Define loss function (criterion) and optimizer
    criterion = get_loss(cfg, device=device)

    # Load checkpoint model
    model_dict = model.state_dict()
    checkpoint_file = args.weights[0]  # args.weights
    logger.info(f"=> loading checkpoint '{checkpoint_file}'")
    checkpoint = torch.load(checkpoint_file)
    checkpoint_dict = checkpoint['state_dict']
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    logger.info(f"=> loaded checkpoint '{checkpoint_file}'")

    model = model.to(device)
    model.gr = 1.0
    model.nc = 1

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
        num_workers=0,  # cfg.WORKERS # Must be 0 or will cause pickling error
        pin_memory=False,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )

    epoch = 0  # special for test

    # Normal Validation    
    da_segment_results, ll_segment_results, detect_results, total_loss, maps, times = validate(
        epoch, cfg, valid_loader, valid_dataset, model, criterion,
        output_dir=final_output_dir, tb_log_dir=tb_log_dir, writer_dict=writer_dict,
        logger=logger, device=device
    )

    msg = ('Test:    Loss({loss:.3f})\n'
           'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n'
           'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n'
           'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'
           'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)').format(
        loss=total_loss, da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],
        ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2],
        p=detect_results[0], r=detect_results[1], map50=detect_results[2], map=detect_results[3],
        t_inf=times[0], t_nms=times[1])

    logger.info(msg)
    
    # Return results for logging
    return {
        'attack_type': attack_type,
        'attack_params': attack_params,
        'defense_type': defense_type,
        'total_loss': total_loss,
        'da_seg_acc': da_segment_results[0],
        'da_seg_iou': da_segment_results[1],
        'da_seg_miou': da_segment_results[2],
        'll_seg_acc': ll_segment_results[0],
        'll_seg_iou': ll_segment_results[1],
        'll_seg_miou': ll_segment_results[2],
        'p': detect_results[0],
        'r': detect_results[1],
        'map50': detect_results[2],
        'map': detect_results[3],
        't_inf': times[0],
        't_nms': times[1]
    }

def main():
    # Parse arguments and update config
    args = parse_args()
    update_config(cfg, args)

    results = []
    seen_combinations = set()

    if not os.path.exists(args.defended_images_dir):
        print(f"Directory does not exist: {args.defended_images_dir}")
        return

    for root, dirs, files in os.walk(args.defended_images_dir):
        for file in files:
            if file.endswith('_metadata.json'):
                metadata_path = os.path.realpath(os.path.join(root, file))
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                attack_params = {
                    'attack_type': metadata['attack_type'],
                    'epsilon': metadata.get('epsilon', None),
                    'num_pixels': metadata.get('num_pixels', None),
                    'channel': metadata.get('channel', None)
                }
                defense_params = metadata['defense_params']

                # Create a unique identifier for each combination
                combination_id = (attack_params['attack_type'], defense_params, attack_params.get('epsilon'), attack_params.get('num_pixels'), attack_params.get('channel'))

                if combination_id in seen_combinations:
                    print(f"Skipping already processed combination: {combination_id}")
                    continue

                seen_combinations.add(combination_id)
                
                print(f"Running validation for {attack_params['attack_type']} attack with {defense_params} defense and parameters {attack_params}")
                result = run_validation(cfg, args, attack_params, defense_params)
                results.append(result)
                print(f"Validation result: {result}")

    if results:
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        df_results.to_csv('validation_results.csv', index=False)

        # Create visualizations
        # Table visualization
        print(df_results)

        # Graph visualization
        df_results.pivot(index='defense_type', columns='attack_type', values='total_loss').plot(kind='bar')
        plt.title('Total Loss for Different Attacks and Defenses')
        plt.xlabel('Defense')
        plt.ylabel('Total Loss')
        plt.xticks(rotation=45)
        plt.legend(title='Attack')
        plt.tight_layout()
        plt.savefig('total_loss_by_attack_and_defense.png')
    else:
        print("No validation results to process.")

if __name__ == "__main__":
    main()
