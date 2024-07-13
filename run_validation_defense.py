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
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

from lib.core.Attacks.FGSM import fgsm_attack, fgsm_attack_with_noise, iterative_fgsm_attack
from lib.core.Attacks.JSMA import calculate_saliency, find_and_perturb_highest_scoring_pixels
from lib.core.Attacks.UAP import uap_sgd_yolop
from lib.core.Attacks.CCP import color_channel_perturbation

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
    parser.add_argument('--early_stop_threshold',
                        help='early stopping threshold for total loss',
                        type=float,
                        default= .65)  
    parser.add_argument('--batch_size',
                        help='number of combinations to process in each batch',
                        type=int,
                        default=10)
    return parser.parse_args()

def apply_attack(valid_loader, model, device, attack_params, criterion, cfg):
    if attack_params['attack_type'] == 'FGSM':
        epsilon = attack_params['epsilon']
        perturbed_images = []
        for img, target, paths, shapes in valid_loader:
            img = img.to(device)
            img.requires_grad = True
            output = model(img)
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()
            img_grad = img.grad.data
            perturbed_image = fgsm_attack(img, epsilon, img_grad)
            perturbed_images.append(perturbed_image)
        return torch.cat(perturbed_images)

    elif attack_params['attack_type'] == 'JSMA':
        saliency_maps = calculate_saliency(model, valid_loader, device, cfg, criterion)
        images = []
        for batch in valid_loader:
            images.extend(batch[0].numpy())
            break
        perturbed_images, _ = find_and_perturb_highest_scoring_pixels(images, saliency_maps, attack_params['num_pixels'], attack_params['jsma_perturbation'], perturbation_type=attack_params['jsma_attack_type'])
        return perturbed_images

    elif attack_params['attack_type'] == 'UAP':
        uap, loss_history = uap_sgd_yolop(model, valid_loader, device, attack_params['uap_max_iterations'], attack_params['uap_eps'], criterion, attack_params['uap_delta'], attack_params['uap_num_classes'], attack_params['uap_targeted'], attack_params['uap_batch_size'])
        images = []
        for batch in valid_loader:
            images.extend(batch[0].numpy())
            break
        perturbed_images = torch.clamp(torch.tensor(images, dtype=torch.float32) + uap.unsqueeze(0), 0, 1)
        return perturbed_images

    elif attack_params['attack_type'] == 'CCP':
        epsilon = attack_params['epsilon']
        channel = attack_params['channel']
        perturbed_images = []
        for img, target, paths, shapes in valid_loader:
            img = img.to(device)
            img.requires_grad = True
            output = model(img)
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()
            img_grad = img.grad.data
            perturbed_image = color_channel_perturbation(img, epsilon, img_grad, channel)
            perturbed_images.append(perturbed_image)
        return torch.cat(perturbed_images)
    return None

def run_validation(cfg, args, attack_params, defense_params, baseline=False):
    if baseline:
        cfg.defrost()
        cfg.DATASET.TEST_SET = 'val'
        cfg.freeze()
        validation_type = 'normal'
        attack_type = 'Baseline'
        defense_type = 'None'
    else:
        attack_type = attack_params['attack_type']
        if defense_params is None:
            cfg.defrost()
            cfg.DATASET.TEST_SET = 'val'
            cfg.freeze()
            validation_type = 'attack'
            defense_type = 'None'
        else:
            cfg.defrost()
            cfg.DATASET.TEST_SET = f'{attack_type}/{defense_params}'
            cfg.freeze()
            validation_type = 'defense'

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'test', attack_type=attack_type, defense_type=defense_type)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    device = select_device(logger, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)) if not cfg.DEBUG else select_device(logger, 'cpu')
    model = get_net(cfg)
    criterion = get_loss(cfg, device=device)

    model_dict = model.state_dict()
    checkpoint_file = args.weights[0]
    logger.info(f"=> loading checkpoint '{checkpoint_file}'")
    checkpoint = torch.load(checkpoint_file)
    checkpoint_dict = checkpoint['state_dict']
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    logger.info(f"=> loaded checkpoint '{checkpoint_file}'")

    model = model.to(device)
    model.gr = 1.0
    model.nc = 1

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([transforms.ToTensor(), normalize])
    )

    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )

    epoch = 0

    if validation_type == 'attack':
        # Apply attack
        perturbed_images = apply_attack(valid_loader, model, device, attack_params, criterion, cfg)

    da_segment_results, ll_segment_results, detect_results, total_loss, maps, times = validate(
        epoch, cfg, valid_loader, valid_dataset, model, criterion,
        output_dir=final_output_dir, tb_log_dir=tb_log_dir, writer_dict=writer_dict,
        logger=logger, device=device, perturbed_images=perturbed_images if validation_type == 'attack' else None
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

def save_results(results, file_name, directory='.'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name)
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)
    print(f"Saved results to {file_path}")

def prioritize_combinations(task_list):
    prioritized_combinations = []
    for attack_params, defense_params in task_list:
        attack_type = attack_params['attack_type']
        if attack_type == 'FGSM':
            if defense_params in ['resizing', 'compression', 'gaussian_blur']:
                prioritized_combinations.insert(0, (attack_params, defense_params))
        elif attack_type == 'JSMA':
            if defense_params in ['resizing', 'compression', 'gaussian_blur']:
                prioritized_combinations.insert(0, (attack_params, defense_params))
        elif attack_type == 'UAP':
            if defense_params in ['resizing', 'compression']:
                prioritized_combinations.insert(0, (attack_params, defense_params))
        elif attack_type == 'CCP':
            if defense_params in ['compression', 'gaussian_blur']:
                prioritized_combinations.insert(0, (attack_params, defense_params))
        else:
            prioritized_combinations.append((attack_params, defense_params))
    return prioritized_combinations

def plot_bar(df, metric, title, y_label, filename):
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x='defense_type', y=metric, hue='attack_type')
    plt.title(title)
    plt.xlabel('Defense Type')
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.legend(title='Attack Type')
    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()

def plot_heatmap(df, metric, title, filename):
    pivot_table = df.pivot_table(index="defense_type", columns="attack_type", values=metric)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
    plt.title(title)
    plt.xlabel('Attack Type')
    plt.ylabel('Defense Type')
    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()

def plot_box(df, metric, title, y_label, filename):
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='defense_type', y=metric, hue='attack_type')
    plt.title(title)
    plt.xlabel('Defense Type')
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.legend(title='Attack Type')
    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()

def plot_line(df, metric, param, title, y_label, filename):
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df, x=param, y=metric, hue='defense_type', style='attack_type', markers=True, dashes=False)
    plt.title(title)
    plt.xlabel(param)
    plt.ylabel(y_label)
    plt.legend(title='Defense Type')
    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()

def main():
    args = parse_args()
    update_config(cfg, args)

    results = []
    skipped_combinations = []
    seen_combinations = set()
    
    save_dir = 'results'  # Specify your desired directory here

    # Create a unique identifier for the current run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if not os.path.exists(args.defended_images_dir):
        print(f"Directory does not exist: {args.defended_images_dir}")
        return

    task_list = []
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

                combination_id = (attack_params['attack_type'], defense_params, attack_params.get('epsilon'), attack_params.get('num_pixels'), attack_params.get('channel'))

                if combination_id in seen_combinations:
                    continue

                seen_combinations.add(combination_id)
                task_list.append((attack_params, defense_params))

    task_list = prioritize_combinations(task_list)

    # Use tqdm for progress bar
    for i in tqdm(range(0, len(task_list), args.batch_size), desc="Validating combinations"):
        batch = task_list[i:i + args.batch_size]
        
        for attack_params, defense_params in batch:
            print(f"\nRunning validation for {attack_params['attack_type']} attack with {defense_params} defense and parameters {attack_params}")
            result = run_validation(cfg, args, attack_params, defense_params)

            # Add epsilon to the result
            result['epsilon'] = attack_params.get('epsilon', None)

            if result['total_loss'] > args.early_stop_threshold:
                print(f"Early stopping: Loss {result['total_loss']} exceeds threshold {args.early_stop_threshold}")
                skipped_combinations.append(result)
                save_results(skipped_combinations, f'skipped_combinations_{timestamp}.csv', save_dir)
                continue

            results.append(result)
            save_results(results, f'validation_results_{timestamp}.csv', save_dir)
            print(f"Validation result: {result}")
            
        print(f"\nthe value of i is {i}\n")
        
        if i == 20:
            print(f"Processed third batch, breaking now.")
            break

    if results:
        save_results(results, f'validation_results_{timestamp}.csv', save_dir)

        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(save_dir, f'validation_results_{timestamp}.csv'), index=False)
        print(df_results)

        # Visualize results
        plot_bar(df_results, 'da_seg_acc', 'Driving Area Segmentation Accuracy', 'Accuracy', os.path.join(save_dir, f'da_seg_acc_bar_{timestamp}.png'))
        plot_heatmap(df_results, 'da_seg_iou', 'Driving Area Segmentation IoU', os.path.join(save_dir, f'da_seg_iou_heatmap_{timestamp}.png'))
        plot_box(df_results, 'll_seg_miou', 'Lane Line Segmentation mIoU', 'mIoU', os.path.join(save_dir, f'll_seg_miou_box_{timestamp}.png'))
        plot_line(df_results, 'map', 'epsilon', 'Mean Average Precision over Epsilon', 'mAP', os.path.join(save_dir, f'map_line_{timestamp}.png'))

        # Optionally, create a heatmap for more detailed analysis
        pivot_table = df_results.pivot_table(index="defense_type", columns="attack_type", values="total_loss")
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, cmap="YlGnBu")
        plt.title('Heatmap of Total Loss for Different Attacks and Defenses')
        plt.xlabel('Attack Type')
        plt.ylabel('Defense Type')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'heatmap_total_loss_{timestamp}.png'))
        plt.show()
    else:
        print(f"No results to show.")
        
    if skipped_combinations:
        save_results(skipped_combinations, f'skipped_combinations_{timestamp}.csv', save_dir)
        df_skipped = pd.DataFrame(skipped_combinations)
        df_skipped.to_csv(os.path.join(save_dir, f'skipped_combinations_{timestamp}.csv'), index=False)
        print("Skipped combinations due to exceeding the threshold:")
        print(df_skipped)

        # Visualize detailed metrics for skipped combinations
        metrics = ['da_seg_acc', 'da_seg_iou', 'da_seg_miou', 'll_seg_acc', 'll_seg_iou', 'll_seg_miou', 'p', 'r', 'map50', 'map']
        for metric in metrics:
            plot_box(df_skipped, metric, f'{metric} for Skipped Combinations', metric, os.path.join(save_dir, f'skipped_{metric}_box_{timestamp}.png'))
            
if __name__ == "__main__":
    main()
