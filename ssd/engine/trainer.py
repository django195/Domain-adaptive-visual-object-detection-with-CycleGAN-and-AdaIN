import collections
import datetime
import logging
import os
import time
import torch
import torch.distributed as dist
from torchvision.utils import save_image

import random 
from PIL import Image
from ssd.engine.inference import do_evaluation
from ssd.utils import dist_util
from ssd.utils.metric_logger import MetricLogger 
from ssd.Adain.test_adain import style_transfer
from ssd.Adain.test_adain import test_transform
from ssd.Adain.test_adain import SetVggDecoder


def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = '{}/{}'.format(prefix, key)
        if isinstance(value, collections.Mapping):
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = dist_util.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def translate_images(vgg,decoder,style_images,content_images,device,probability):
  output_images = []
  p=100-probability
  for image in content_images:
    #if random.randint(0,9)> 1: 
    #if random.randint(0,99)>69 :
    if random.randint(1, 100) > p:
      content = image.to(device).unsqueeze(0)
      n=random.randint(0,len(style_images)-1)
      style = style_images[n].to(device).unsqueeze(0)
      #vgg.to(device)
      #decoder.to(device)
      output=style_transfer(vgg,decoder,content,style)
      #vgg.cpu()
      #decoder.cpu()
      #content.cpu()
      #style.cpu()
      output=torch.nn.functional.interpolate(output,size=(300,300))
      output=output.to(device).squeeze(0)
      output_images.append(output)
    else:
      image=image.to(device)
      output_images.append(image)
  return torch.stack(output_images)

def do_train(cfg, model,
             data_loader,
             optimizer,
             scheduler,
             checkpointer,
             device,
             arguments,
             args,
             style_loader=False):
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")
    meters = MetricLogger()
    
    model.train()
    save_to_disk = dist_util.get_rank() == 0 
    if args.use_tensorboard and save_to_disk:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from tensorboardX import SummaryWriter
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    
    if style_loader:
      vgg, decoder = SetVggDecoder(device)
      style_iter=iter(style_loader)

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        iteration = iteration + 1
        arguments["iteration"] = iteration

        #images = images.to(device)
        if args.adain:
          #se arriva alla fine deve ricominciare da capo
          try:
            style_images = next(style_iter) #take a new batch of Clipart in the format [Images,Target,ImageId]
          except StopIteration:
            style_iter = iter(style_loader)
            style_images = next(style_iter)
          images=translate_images(vgg,decoder,style_images[0],images,device,args.probability)
        else:  
          images=images.to(device)

        targets = targets.to(device)
        loss_dict = model(images, targets=targets)
        loss = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(total_loss=losses_reduced, **loss_dict_reduced)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time)
        if iteration % args.log_step == 0:
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            if device == "cuda":
                logger.info(
                    meters.delimiter.join([
                        "iter: {iter:06d}",
                        "lr: {lr:.8f}",
                        '{meters}',
                        "eta: {eta}",
                        'mem: {mem}M',
                    ]).format(
                        iter=iteration,
                        lr=optimizer.param_groups[0]['lr'],
                        meters=str(meters),
                        eta=eta_string,
                        mem=round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
                    )
                )
            else:
                logger.info(
                    meters.delimiter.join([
                        "iter: {iter:06d}",
                        "lr: {lr:.8f}",
                        '{meters}',
                        "eta: {eta}",
                    ]).format(
                        iter=iteration,
                        lr=optimizer.param_groups[0]['lr'],
                        meters=str(meters),
                        eta=eta_string,
                    )
                )
            if summary_writer:
                global_step = iteration
                summary_writer.add_scalar('losses/total_loss', losses_reduced, global_step=global_step)
                for loss_name, loss_item in loss_dict_reduced.items():
                    summary_writer.add_scalar('losses/{}'.format(loss_name), loss_item, global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if iteration % args.save_step == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)

        if args.eval_step > 0 and iteration % args.eval_step == 0 and not iteration == max_iter:
            eval_results = do_evaluation(cfg, model, distributed=args.distributed, iteration=iteration)
            if dist_util.get_rank() == 0 and summary_writer:
                for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                    write_metric(eval_result['metrics'], 'metrics/' + dataset, summary_writer, iteration)
            model.train()

    checkpointer.save("model_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return model
