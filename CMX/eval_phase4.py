#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre
from phase4_metrics import save_phase4_metrics

os.environ['MASTER_ADDR']='127.0.0.1'
os.environ['MASTER_PORT']='29500'
print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
logger=get_logger()

class SegEvaluator(Evaluator):
    def func_per_iteration(self,data,device):
        img=data['data']
        label=data['label']
        modal_x=data['modal_x']
        name=data['fn']
        pred=self.sliding_eval_rgbX(img,modal_x,config.eval_crop_size,config.eval_stride_rate,device)
        hist_tmp,labeled_tmp,correct_tmp=hist_info(config.num_classes,pred,label)
        results_dict={'hist':hist_tmp,'labeled':labeled_tmp,'correct':correct_tmp}

        # Se conserva el guardado RAW si se solicita.
        # No es necesario para calcular las métricas de Fase 4.
        if self.save_path is not None:
            ensure_dir(self.save_path)
            fn=name+'.png'
            cv2.imwrite(os.path.join(self.save_path,fn),pred)
            logger.info('Save the image '+fn)

        if self.show_image:
            colors=self.dataset.get_class_colors
            image=img
            clean=np.zeros(label.shape)
            comp_img=show_img(colors,config.background,image,clean,label,pred)
            cv2.imshow('comp_image',comp_img)
            cv2.waitKey(0)
        return results_dict

    def compute_metric(self,results):
        hist=np.zeros((config.num_classes,config.num_classes))
        correct=0
        labeled=0
        for d in results:
            hist+=d['hist']
            correct+=d['correct']
            labeled+=d['labeled']

        # Cálculo ORIGINAL del repositorio.
        iou,mean_IoU,class_acc,freq_IoU,mean_pixel_acc,pixel_acc=compute_score(hist,correct,labeled)
        result_line=print_iou(
            iou,freq_IoU,mean_pixel_acc,pixel_acc,
            dataset.class_names,show_no_back=False
        )

        # NUEVO: solo postprocesa la misma matriz hist.
        metrics_dir=getattr(self,'phase4_metrics_dir',None)
        subset_name=getattr(self,'phase4_subset_name','test')
        if metrics_dir:
            existing={
                'mean_IoU':None if not np.isfinite(mean_IoU) else float(mean_IoU),
                'freq_IoU':None if not np.isfinite(freq_IoU) else float(freq_IoU),
                'mean_pixel_acc':None if not np.isfinite(mean_pixel_acc) else float(mean_pixel_acc),
                'pixel_acc':None if not np.isfinite(pixel_acc) else float(pixel_acc)
            }
            save_phase4_metrics(
                hist=hist,
                class_names=dataset.class_names,
                output_dir=metrics_dir,
                subset_name=subset_name,
                existing_metrics=existing
            )
            logger.info('Phase 4 metrics saved in: {}'.format(metrics_dir))
        return result_line

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-e','--epochs',default='last',type=str)
    parser.add_argument('-d','--devices',default='0',type=str)
    parser.add_argument('-v','--verbose',default=False,action='store_true')
    parser.add_argument('--show_image','-s',default=False,action='store_true')
    parser.add_argument('--save_path','-p',default=None)

    # Fase 4
    parser.add_argument('--eval_source',default=None,
                        help='Lista de evaluación. Si se omite, usa config.eval_source.')
    parser.add_argument('--metrics_dir',default=None,
                        help='Carpeta de salida para Precision/Recall/F1/confusion matrix.')
    parser.add_argument('--subset_name',default='test',
                        help='Nombre del subconjunto: full/day/tunnel/night.')

    args=parser.parse_args()
    all_dev=parse_devices(args.devices)
    eval_source=args.eval_source if args.eval_source else config.eval_source

    print("="*64)
    print("FASE 4 — EVALUACION EXTENDIDA")
    print("Eval source :",eval_source)
    print("Subset      :",args.subset_name)
    print("Metrics dir :",args.metrics_dir)
    print("="*64)

    network=segmodel(cfg=config,criterion=None,norm_layer=nn.BatchNorm2d)
    data_setting={
        'rgb_root':config.rgb_root_folder,
        'rgb_format':config.rgb_format,
        'gt_root':config.gt_root_folder,
        'gt_format':config.gt_format,
        'transform_gt':config.gt_transform,
        'x_root':config.x_root_folder,
        'x_format':config.x_format,
        'x_single_channel':config.x_is_single_channel,
        'class_names':config.class_names,
        'train_source':config.train_source,
        'eval_source':eval_source
    }
    val_pre=ValPre()
    dataset=RGBXDataset(data_setting,'val',val_pre)

    with torch.no_grad():
        segmentor=SegEvaluator(
            dataset,config.num_classes,config.norm_mean,config.norm_std,network,
            config.eval_scale_array,config.eval_flip,all_dev,args.verbose,
            args.save_path,args.show_image
        )
        segmentor.phase4_metrics_dir=args.metrics_dir
        segmentor.phase4_subset_name=args.subset_name
        segmentor.run(
            config.checkpoint_dir,args.epochs,
            config.val_log_file,config.link_val_log_file
        )
