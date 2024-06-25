import torch
import os, sys  
import numpy as np
import pandas as pd
from tqdm import tqdm  
from pathlib import Path
import json
import random
import cv2
import math
import datetime
from torch.utils.data import DataLoader  
import matplotlib.pyplot as plt

from lib.core.evaluate import ConfusionMatrix,SegmentationMetric
from lib.core.function import AverageMeter
from lib.core.general import non_max_suppression,check_img_size,scale_coords,xyxy2xywh,xywh2xyxy,box_iou,coco80_to_coco91_class,plot_images,ap_per_class,output_to_target
from lib.utils.utils import time_synchronized
from lib.utils import plot_img_and_mask,plot_one_box,show_seg_result

# JSMA Helper Functions 
def calculate_saliency(model, valid_loader, device, config, criterion):
    """
    Calculates the saliency maps for images from a validation data loader using a given model.

    Args:
        model (torch.nn.Module): The model used for computing the outputs.
        valid_loader (DataLoader): DataLoader containing the validation dataset.
        device (torch.device): The device (GPU/CPU) on which to perform computations.
        config (object): Configuration object containing runtime settings such as DEBUG mode.
        criterion (function): Loss function used to compute the error between predictions and targets.

    Returns:
        numpy.ndarray: An array of saliency maps for the first batch of images from the valid_loader.
    """
    
    model.eval()
    saliency_maps = []
    import time
    start_t = time.time()
    for batch_i, (img, target, paths, shapes) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        
        if not config.DEBUG:
            img = img.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
            nb, _, height, width = img.shape  # batch size, channel, height, width
            
        # Enable gradient computation for the input image
        img.requires_grad = True
        
        # Forward pass
        det_out, da_seg_out, ll_seg_out = model(img)
        inf_out, train_out = det_out 
        
        # Calculate loss
        total_loss, head_losses = criterion((train_out, da_seg_out, ll_seg_out), target, shapes, model)
        total_loss.backward()
        
        # Compute saliency map for each image in the batch
        #saliency_maps = img.grad.abs().max(dim=1, keepdim=True)[0].squeeze().detach().cpu().numpy()
        #saliency_maps = img.grad.abs().max(dim=3, keepdim=True)[0].detach().cpu().numpy()
        saliency_maps_batch = img.grad.abs().detach().numpy()
        saliency_maps.append(saliency_maps_batch)
        
        if batch_i == 0:
            break
    end_t = time.time()
    print("total time in calculating saliency maps {}s".format(end_t - start_t))
    saliency_maps = np.concatenate(saliency_maps, axis=0)
    
    # Plot and save the saliency maps and original images
    img = img.detach().cpu().numpy()  # Move the original images back to CPU and convert to numpy
  
    return saliency_maps


def find_and_perturb_highest_scoring_pixels(images, saliency_maps, num_pixels_to_perturb, perturbation_value, perturbation_type='add'):
    """
    Perturbs the highest scoring pixels in images based on their saliency maps to investigate the effect on model predictions.

    Args:
        images (list of numpy.ndarray): The original images.
        saliency_maps (list of numpy.ndarray): The saliency maps of the images.
        num_pixels_to_perturb (int): Number of top pixels to perturb in each map.
        perturbation_value (float): The value to add to the top pixels in the images.
        perturbation_type (str): Type of perturbation ('add', 'set', 'noise').

    Returns:
        tuple: Contains the perturbed images tensor and the coordinates of the perturbed pixels.
    """
    perturbed_images = []
    all_top_coords = []
    

    for image, saliency_map in zip(images, saliency_maps):
        # Flatten the saliency map to get the indices of the top pixels
        flat_indices = np.argsort(saliency_map.flatten())[::-1]
        top_indices = flat_indices[:num_pixels_to_perturb]

        # Convert the flat indices back to 2D coordinates (y, x)
        top_coords = np.unravel_index(top_indices, saliency_map.shape)
        y_coords, x_coords = top_coords[1], top_coords[2]

        # Create a copy of the image to perturb
        perturbed_image = image.copy()
        
        # Apply perturbation to the top pixels
        for coord in zip(*top_coords):
            if perturbation_type == 'add':
                perturbed_image[coord] += perturbation_value
            elif perturbation_type == 'set':
                perturbed_image[coord] = perturbation_value
            elif perturbation_type == 'noise':
                perturbed_image[coord] += np.random.normal(perturbation_value)

        # Ensure pixel values are within valid range if necessary
        perturbed_image = np.clip(perturbed_image, 0, 1)

        # Convert the perturbed image to a tensor and add a batch dimension
        perturbed_image_tensor = torch.tensor(perturbed_image, dtype=torch.float32).unsqueeze(0)
        perturbed_images.append(perturbed_image_tensor)

        all_top_coords.append(top_coords)
        
    return torch.cat(perturbed_images, dim=0), all_top_coords

def validateJSMA(epoch,config, val_loader, val_dataset, model, criterion, output_dir,
tb_log_dir, perturbed_images, experiment_number, writer_dict=None, logger=None, device='cpu', rank=-1):
    """
    validata

    Inputs:
    - config: configurations 
    - train_loader: loder for data
    - model: 
    - criterion: (function) calculate all the loss, return 
    - writer_dict: 

    Return:
    None
    """
    # setting
    max_stride = 32
    weights = None

    save_dir = os.path.join(output_dir, f'visualization_exp_{experiment_number}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # print(save_dir)
    _, imgsz = [check_img_size(x, s=max_stride) for x in config.MODEL.IMAGE_SIZE] #imgsz is multiple of max_stride
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(config.GPUS)
    test_batch_size = config.TEST.BATCH_SIZE_PER_GPU * len(config.GPUS)
    training = False
    is_coco = False #is coco dataset
    save_conf=False # save auto-label confidences
    verbose=False
    save_hybrid=False
    log_imgs,wandb = min(16,100), None

    nc = 1
    iouv = torch.linspace(0.5,0.95,10).to(device)     #iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    try:
        import wandb
    except ImportError:
        wandb = None
        log_imgs = 0

    seen =  0 
    confusion_matrix = ConfusionMatrix(nc=model.nc) #detector confusion matrix
    da_metric = SegmentationMetric(config.num_seg_class) #segment confusion matrix    
    ll_metric = SegmentationMetric(2) #segment confusion matrix

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    coco91class = coco80_to_coco91_class()
    
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t_inf, t_nms = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    
    losses = AverageMeter()

    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()

    T_inf = AverageMeter()
    T_nms = AverageMeter()

    # switch to evaluation mode
    model.eval()
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    
    img = perturbed_images.to(device, non_blocking=True)

    for batch_i, (img_throw_away, target, paths, shapes) in tqdm(enumerate(val_loader), total=len(val_loader)):
        if not config.DEBUG:

            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
            nb, _, height, width = img.shape
            
       


        with torch.no_grad():
            pad_w, pad_h = shapes[0][1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            ratio = shapes[0][1][0][0]

            t = time_synchronized()
            det_out, da_seg_out, ll_seg_out = model(img)
            t_inf = time_synchronized() - t
            if batch_i > 0:
                T_inf.update(t_inf/img.size(0),img.size(0))

            inf_out, train_out = det_out

            #driving area segment evaluation
            _,da_predict=torch.max(da_seg_out, 1)
            _,da_gt=torch.max(target[1], 1)
            da_predict = da_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            da_gt = da_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            da_metric.reset()
            da_metric.addBatch(da_predict.cpu(), da_gt.cpu())
            da_acc = da_metric.pixelAccuracy()
            da_IoU = da_metric.IntersectionOverUnion()
            da_mIoU = da_metric.meanIntersectionOverUnion()

            da_acc_seg.update(da_acc,img.size(0))
            da_IoU_seg.update(da_IoU,img.size(0))
            da_mIoU_seg.update(da_mIoU,img.size(0))

            #lane line segment evaluation
            _,ll_predict=torch.max(ll_seg_out, 1)
            _,ll_gt=torch.max(target[2], 1)
            ll_predict = ll_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            ll_gt = ll_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            ll_metric.reset()
            ll_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
            ll_acc = ll_metric.lineAccuracy()
            ll_IoU = ll_metric.IntersectionOverUnion()
            ll_mIoU = ll_metric.meanIntersectionOverUnion()

            ll_acc_seg.update(ll_acc,img.size(0))
            ll_IoU_seg.update(ll_IoU,img.size(0))
            ll_mIoU_seg.update(ll_mIoU,img.size(0))
            
            total_loss, head_losses = criterion((train_out,da_seg_out, ll_seg_out), target, shapes, model)   # Compute loss
            losses.update(total_loss.item(), img.size(0))

            #NMS
            t = time_synchronized()
            target[0][:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [target[0][target[0][:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            output = non_max_suppression(inf_out, conf_thres= config.TEST.NMS_CONF_THRESHOLD, iou_thres=config.TEST.NMS_IOU_THRESHOLD, labels=lb)
            
            
            t_nms = time_synchronized() - t
            
            if batch_i > 0:
                T_nms.update(t_nms/img.size(0),img.size(0))

            if config.TEST.PLOTS:
                if batch_i == 0:
                    for i in range(test_batch_size):
                        img_test = cv2.imread(paths[i])
                        da_seg_mask = da_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        da_seg_mask = torch.nn.functional.interpolate(da_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, da_seg_mask = torch.max(da_seg_mask, 1)

                        da_gt_mask = target[1][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        da_gt_mask = torch.nn.functional.interpolate(da_gt_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, da_gt_mask = torch.max(da_gt_mask, 1)

                        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
                        da_gt_mask = da_gt_mask.int().squeeze().cpu().numpy()
                        # seg_mask = seg_mask > 0.5
                        # plot_img_and_mask(img_test, seg_mask, i,epoch,save_dir)
                        img_test1 = img_test.copy()
                        _ = show_seg_result(img_test, da_seg_mask, i,epoch,save_dir, attack_type = "jsma")
                        # _ = show_seg_result(img_test1, da_gt_mask, i, epoch, save_dir, is_gt=True, attack_type="jsma")

                        img_ll = cv2.imread(paths[i])
                        ll_seg_mask = ll_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, ll_seg_mask = torch.max(ll_seg_mask, 1)

                        ll_gt_mask = target[2][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        ll_gt_mask = torch.nn.functional.interpolate(ll_gt_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, ll_gt_mask = torch.max(ll_gt_mask, 1)

                        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
                        ll_gt_mask = ll_gt_mask.int().squeeze().cpu().numpy()
                        # seg_mask = seg_mask > 0.5
                        # plot_img_and_mask(img_test, seg_mask, i,epoch,save_dir)
                        img_ll1 = img_ll.copy()
                        _ = show_seg_result(img_ll, ll_seg_mask, i,epoch,save_dir, is_ll=True, attack_type = "jsma")
                        # _ = show_seg_result(img_ll1, ll_gt_mask, i, epoch, save_dir, is_ll=True, is_gt=True, attack_type = "jsma")

                        img_det = cv2.imread(paths[i])
                        img_gt = img_det.copy()
                        det = output[i].clone()
                        if len(det):
                            det[:,:4] = scale_coords(img[i].shape[1:],det[:,:4],img_det.shape).round()
                        for *xyxy,conf,cls in reversed(det):
                            #print(cls)
                            label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=3)
                        cv2.imwrite(save_dir+"/batch_{}_{}_jsma_det_pred.png".format(epoch,i),img_det)

                        labels = target[0][target[0][:, 0] == i, 1:]
                        # print(labels)
                        labels[:,1:5]=xywh2xyxy(labels[:,1:5])
                        if len(labels):
                            labels[:,1:5]=scale_coords(img[i].shape[1:],labels[:,1:5],img_gt.shape).round()
                    '''    for cls,x1,y1,x2,y2 in labels:
                            #print(names)
                            #print(cls)
                            label_det_gt = f'{names[int(cls)]}'
                            xyxy = (x1,y1,x2,y2)
                            plot_one_box(xyxy, img_gt , label=label_det_gt, color=colors[int(cls)], line_thickness=3)
                         cv2.imwrite(save_dir+"/batch_{}_{}_jsma_det_gt.png".format(epoch,i),img_gt)
                            This is GT stuff '''
        # Statistics per image
        # output([xyxy,conf,cls])
        # target[0] ([img_id,cls,xyxy])
        for si, pred in enumerate(output):
            labels = target[0][target[0][:, 0] == si, 1:]     # all object in one image 
            nl = len(labels)    # num of object
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if config.TEST.SAVE_TXT:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging
            if config.TEST.PLOTS and len(wandb_images) < log_imgs:
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls),
                             "box_caption": "%s %.3f" % (names[cls], conf),
                             "scores": {"class_score": conf},
                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Append to pycocotools JSON dictionary
            if config.TEST.SAVE_JSON:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})


            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if config.TEST.PLOTS:
                    confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):                    
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # n*m  n:pred  m:label
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        if config.TEST.PLOTS and batch_i < 3:
            f = save_dir +'/'+ f'test_batch{batch_i}_labels.jpg'  # labels
            #Thread(target=plot_images, args=(img, target[0], paths, f, names), daemon=True).start()
            f = save_dir +'/'+ f'test_batch{batch_i}_pred.jpg'  # predictions
            #Thread(target=plot_images, args=(img, output_to_target(output), paths, f, names), daemon=True).start()

        if batch_i == 0:
            break
        
    # Compute statistics
    # stats : [[all_img_correct]...[all_img_tcls]]
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy  zip(*) :unzip

    map70 = None
    map75 = None
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
        ap50, ap70, ap75,ap = ap[:, 0], ap[:,4], ap[:,5],ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map70, map75, map = p.mean(), r.mean(), ap50.mean(), ap70.mean(),ap75.mean(),ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    #print(map70)
    #print(map75)

    # Print results per class
    if (verbose or (nc <= 20 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t_inf, t_nms, t_inf + t_nms)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if config.TEST.PLOTS:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb and wandb.run:
            wandb.log({"Images": wandb_images})
            wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]})

    # Save JSON
    if config.TEST.SAVE_JSON and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in val_loader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if config.TEST.SAVE_TXT else ''
        print(f"Results saved to {save_dir}{s}")
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg,ll_IoU_seg.avg,ll_mIoU_seg.avg)

    # print(da_segment_result)
    # print(ll_segment_result)
    detect_result = np.asarray([mp, mr, map50, map])
    # print('mp:{},mr:{},map50:{},map:{}'.format(mp, mr, map50, map))
    #print segmet_result
    t = [T_inf.avg, T_nms.avg]
    
    return da_segment_result, ll_segment_result, detect_result, losses.avg, maps, t

def run_jsma_experiments(model, valid_loader, device, config, criterion, perturbation_params, final_output_directory):
    results = []
    experiment_number = 0

    for num_pixels, perturb_value, attack_type in perturbation_params:
        experiment_number+=1
        # Calculate saliency maps
        saliency_maps = calculate_saliency(model, valid_loader, device, config, criterion)
        
        # Perturb images
        # Extract images from the data loader
        images = []
        for batch in valid_loader:
            images.extend(batch[0].numpy())
            if 0 == 0 :
                print("Breaking...")
                break
            
        perturbed_images, _ = find_and_perturb_highest_scoring_pixels(
            images, saliency_maps, num_pixels, perturb_value, perturbation_type=attack_type
        )

        # Validate the model with perturbed images
        da_segment_result, ll_segment_result, detect_result, loss_avg, maps, t = validateJSMA(
            epoch=0,
            config=config,
            val_loader=valid_loader,
            val_dataset=valid_loader.dataset,
            model=model,
            criterion=criterion,
            output_dir= final_output_directory,
            tb_log_dir="log",
            perturbed_images=perturbed_images,
            experiment_number=experiment_number,
            device=device
        )

        # Store the results
        results.append({
            "num_pixels": num_pixels,
            "perturb_value": perturb_value,
            "attack_type": attack_type,
            "da_acc_seg": da_segment_result[0],
            "da_IoU_seg": da_segment_result[1],
            "da_mIoU_seg": da_segment_result[2],
            "ll_acc_seg": ll_segment_result[0],
            "ll_IoU_seg": ll_segment_result[1],
            "ll_mIoU_seg": ll_segment_result[2],
            # "detect_result": detect_result,
            "loss_avg": loss_avg,
            # "maps": maps,
            # "time": t
        })

    return pd.DataFrame(results)