import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import cv2
from torch.utils.data import DataLoader

from lib.core.evaluate import ConfusionMatrix, SegmentationMetric
from lib.core.function import AverageMeter
from lib.core.general import non_max_suppression, check_img_size, scale_coords, xyxy2xywh, xywh2xyxy, box_iou, coco80_to_coco91_class, plot_images, ap_per_class, output_to_target
from lib.utils.utils import time_synchronized
from lib.utils import plot_img_and_mask, plot_one_box, show_seg_result

def uap_sgd_yolop(model, valid_loader, device, nb_epoch, eps, criterion, step_decay, beta=12, y_target=None, loss_fn=None, layer_name=None, uap_init=None):
    """
    Universal Adversarial Perturbation (UAP) via Stochastic Gradient Descent (SGD) for YOLOP

    Args:
        model (torch.nn.Module): The YOLOP model.
        valid_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device (GPU/CPU) on which to perform computations.
        nb_epoch (int): Number of optimization epochs.
        eps (float): Maximum perturbation value (L-infinity norm).
        beta (float, optional): Clamping value. Default is 12.
        step_decay (float, optional): Decay rate for the step size. Default is 0.8.
        y_target (int, optional): Target class label for Targeted UAP variation. Default is None.
        loss_fn (callable, optional): Custom loss function (default is CrossEntropyLoss).
        layer_name (str, optional): Target layer name for layer maximization attack. Default is None.
        uap_init (torch.Tensor, optional): Custom perturbation to start from (default is random vector with pixel values {-eps, eps}).

    Returns:
        torch.Tensor: Adversarial perturbation.
        list: Losses per iteration.
    """
    if uap_init is None:
        delta = torch.rand((1, 3, 640, 640), device=device) * 2 * eps - eps
    else:
        delta = uap_init.to(device)
        
    print(F"The eps {eps}\n")
    print(F"The stepdecay {step_decay}")


    delta.requires_grad = True
    eps_step = eps * step_decay

    losses = []
    
    for epoch in range(nb_epoch):
        for batch_i, (img, target, paths, shapes) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            img = img.to(device, non_blocking=True)
            target = [tgt.to(device) for tgt in target]
            
            # Apply perturbation
            # Get shapes 
            print(f"{img.shape}")
            print(f"{delta.shape}")

                  
            perturbed_img = img + delta
            perturbed_img = torch.clamp(perturbed_img, 0, 1)
            
            # Forward pass
            det_out, da_seg_out, ll_seg_out = model(perturbed_img)
            inf_out, train_out = det_out 
            
                
            total_loss, head_losses = criterion((train_out, da_seg_out, ll_seg_out), target, shapes, model)
                    
            losses.append(total_loss.item())
            model.zero_grad()
            total_loss.backward()
            
            data_grad = delta.grad
            
            # Collect the sign of the gradients
            sign_data_grad = data_grad.sign()

            delta = delta + sign_data_grad * eps_step
            
            delta = torch.clamp(delta, -eps, eps)
            delta.grad.data.zero_()

        # Decay step size
        eps_step *= step_decay
    
    if layer_name is not None:
        handle.remove()  # release hook
        
    return delta.detach(), losses

def validate_uap(epoch, config, val_loader, model, criterion, output_dir, tb_log_dir, perturbed_images, experiment_number, device='cpu'):
    max_stride = 32
    weights = None

    save_dir = os.path.join(output_dir, f'visualization_exp_{experiment_number}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    _, imgsz = [check_img_size(x, s=max_stride) for x in config.MODEL.IMAGE_SIZE]
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(config.GPUS)
    training = False
    is_coco = False
    save_conf = False
    verbose = False
    save_hybrid = False
    log_imgs, wandb = min(16, 100), None

    nc = 1
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=model.nc)
    da_metric = SegmentationMetric(config.num_seg_class)
    ll_metric = SegmentationMetric(2)

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
                T_inf.update(t_inf / img.size(0), img.size(0))

            inf_out, train_out = det_out

            _, da_predict = torch.max(da_seg_out, 1)
            _, da_gt = torch.max(target[1], 1)
            da_predict = da_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            da_gt = da_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            da_metric.reset()
            da_metric.addBatch(da_predict.cpu(), da_gt.cpu())
            da_acc = da_metric.pixelAccuracy()
            da_IoU = da_metric.IntersectionOverUnion()
            da_mIoU = da_metric.meanIntersectionOverUnion()

            da_acc_seg.update(da_acc, img.size(0))
            da_IoU_seg.update(da_IoU, img.size(0))
            da_mIoU_seg.update(da_mIoU, img.size(0))

            _, ll_predict = torch.max(ll_seg_out, 1)
            _, ll_gt = torch.max(target[2], 1)
            ll_predict = ll_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            ll_gt = ll_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            ll_metric.reset()
            ll_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
            ll_acc = ll_metric.lineAccuracy()
            ll_IoU = ll_metric.IntersectionOverUnion()
            ll_mIoU = ll_metric.meanIntersectionOverUnion()

            ll_acc_seg.update(ll_acc, img.size(0))
            ll_IoU_seg.update(ll_IoU, img.size(0))
            ll_mIoU_seg.update(ll_mIoU, img.size(0))

            total_loss, head_losses = criterion((train_out, da_seg_out, ll_seg_out), target, shapes, model)
            losses.update(total_loss.item(), img.size(0))

            t = time_synchronized()
            target[0][:, 2:] *= torch.Tensor([width, height, width, height]).to(device)
            lb = [target[0][target[0][:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
            output = non_max_suppression(inf_out, conf_thres=config.TEST.NMS_CONF_THRESHOLD, iou_thres=config.TEST.NMS_IOU_THRESHOLD, labels=lb)

            t_nms = time_synchronized() - t

            if batch_i > 0:
                T_nms.update(t_nms / img.size(0), img.size(0))

            if config.TEST.PLOTS:
                if batch_i == 0:
                    for i in range(batch_size):
                        img_test = cv2.imread(paths[i])
                        da_seg_mask = da_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        da_seg_mask = torch.nn.functional.interpolate(da_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, da_seg_mask = torch.max(da_seg_mask, 1)

                        da_gt_mask = target[1][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        da_gt_mask = torch.nn.functional.interpolate(da_gt_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, da_gt_mask = torch.max(da_gt_mask, 1)

                        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
                        da_gt_mask = da_gt_mask.int().squeeze().cpu().numpy()
                        img_test1 = img_test.copy()
                        _ = show_seg_result(img_test, da_seg_mask, i, epoch, save_dir, attack_type="uap")
                        
                        img_ll = cv2.imread(paths[i])
                        ll_seg_mask = ll_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, ll_seg_mask = torch.max(ll_seg_mask, 1)

                        ll_gt_mask = target[2][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        ll_gt_mask = torch.nn.functional.interpolate(ll_gt_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, ll_gt_mask = torch.max(ll_gt_mask, 1)

                        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
                        ll_gt_mask = ll_gt_mask.int().squeeze().cpu().numpy()
                        img_ll1 = img_ll.copy()
                        _ = show_seg_result(img_ll, ll_seg_mask, i, epoch, save_dir, is_ll=True, attack_type="uap")

                        img_det = cv2.imread(paths[i])
                        img_gt = img_det.copy()
                        det = output[i].clone()
                        if len(det):
                            det[:, :4] = scale_coords(img[i].shape[1:], det[:, :4], img_det.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img_det, label=label_det_pred, color=colors[int(cls)], line_thickness=3)
                        cv2.imwrite(save_dir + f"/batch_{epoch}_{i}_uap_det_pred.png", img_det)

                        labels = target[0][target[0][:, 0] == i, 1:]
                        labels[:, 1:5] = xywh2xyxy(labels[:, 1:5])
                        if len(labels):
                            labels[:, 1:5] = scale_coords(img[i].shape[1:], labels[:, 1:5], img_gt.shape).round()

            for si, pred in enumerate(output):
                labels = target[0][target[0][:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []
                path = Path(paths[si])
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

                if config.TEST.SAVE_TXT:
                    gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]
                    for *xyxy, conf, cls in predn.tolist():
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if config.TEST.PLOTS and len(wandb_images) < log_imgs:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
                    wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

                if config.TEST.SAVE_JSON:
                    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                    box = xyxy2xywh(predn[:, :4])
                    box[:, :2] -= box[:, 2:] / 2
                    for p, b in zip(pred.tolist(), box.tolist()):
                        jdict.append({'image_id': image_id,
                                      'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5)})

                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    detected = []
                    tcls_tensor = labels[:, 0]
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                    if config.TEST.PLOTS:
                        confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)

                        if pi.shape[0]:
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv
                                    if len(detected) == nl:
                                        break

                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            if config.TEST.PLOTS and batch_i < 3:
                f = save_dir + '/' + f'test_batch{batch_i}_labels.jpg'
                f = save_dir + '/' + f'test_batch{batch_i}_pred.jpg'

            if batch_i == 0:
                break

    stats = [np.concatenate(x, 0) for x in zip(*stats)]

    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
        ap50, ap70, ap75, ap = ap[:, 0], ap[:, 4], ap[:, 5], ap.mean(1)
        mp, mr, map50, map70, map75, map = p.mean(), r.mean(), ap50.mean(), ap70.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
    else:
        nt = torch.zeros(1)

    pf = '%20s' + '%12.3g' * 6
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    if (verbose or (nc <= 20 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    t = tuple(x / seen * 1E3 for x in (t_inf, t_nms, t_inf + t_nms)) + (imgsz, imgsz, batch_size)
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    if config.TEST.PLOTS:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    if config.TEST.SAVE_JSON and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''
        anno_json = '../coco/annotations/instances_val2017.json'
        pred_json = str(save_dir / f"{w}_predictions.json")
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in val_loader.dataset.img_files]
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if config.TEST.SAVE_TXT else ''
        print(f"Results saved to {save_dir}{s}")
    model.float()
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg, ll_IoU_seg.avg, ll_mIoU_seg.avg)

    detect_result = np.asarray([mp, mr, map50, map])
    t = [T_inf.avg, T_nms.avg]

    return da_segment_result, ll_segment_result, detect_result, losses.avg, maps, t

def run_uap_experiments(model, valid_loader, device, config, criterion, uap_params, final_output_directory):
    results = []
    experiment_number = 0

    for nb_epoch, eps, step_decay, y_target, layer_name, beta in uap_params:
        
        print(F"EXP The stepdecay {step_decay}")
        
        experiment_number += 1
        uap, loss_history = uap_sgd_yolop(model, valid_loader, device, nb_epoch, eps, beta, step_decay, y_target, None, layer_name)

        images = []
        for batch in valid_loader:
            images.extend(batch[0].numpy())
            if len(images) >= len(valid_loader.dataset):
                break

        perturbed_images = torch.clamp(torch.tensor(images, dtype=torch.float32) + uap.unsqueeze(0), 0, 1)

        da_segment_result, ll_segment_result, detect_result, loss_avg, maps, t = validate_uap(
            epoch=0,
            config=config,
            val_loader=valid_loader,
            model=model,
            criterion=criterion,
            output_dir=final_output_directory,
            tb_log_dir="log",
            perturbed_images=perturbed_images,
            experiment_number=experiment_number,
            device=device
        )

        results.append({
            "nb_epoch": nb_epoch,
            "eps": eps,
            "step_decay": step_decay,
            "y_target": y_target,
            "layer_name": layer_name,
            "beta": beta,
            "da_acc_seg": da_segment_result[0],
            "da_IoU_seg": da_segment_result[1],
            "da_mIoU_seg": da_segment_result[2],
            "ll_acc_seg": ll_segment_result[0],
            "ll_IoU_seg": ll_segment_result[1],
            "ll_mIoU_seg": ll_segment_result[2],
            "loss_avg": loss_avg,
        })

    return pd.DataFrame(results)
