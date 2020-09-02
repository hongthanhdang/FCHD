# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import HeadDataset
from data.preprocess import Rescale, Normalize, inverse_normalize
from config import cfg
# from utils.visualize import visdom_bbox
import numpy as np
import os
from networks.detector import HeadDetector
from trainer import Trainer
import matplotlib.pyplot as plt
from utils import tools
import time


def train():
    # Load data
    train_annots_path = os.path.join(cfg.BKDATASET_DIR, cfg.BK_PLUS_BRAINWASH_TRAIN_ANNOTS_FILE)
    val_annots_path = os.path.join(cfg.BKDATASET_DIR, cfg.BKVAL_ANNOTS_FILE)
    transform = transforms.Compose([Rescale(), Normalize()])
    train_dataset = HeadDataset(cfg.BKDATASET_DIR, train_annots_path, transform)
    val_dataset = HeadDataset(cfg.BKDATASET_DIR, val_annots_path, transform)

    # train_annots_path = os.path.join(cfg.DATASET_DIR, cfg.TRAIN_ANNOTS_FILE)
    # val_annots_path = os.path.join(cfg.DATASET_DIR, cfg.VAL_ANNOTS_FILE)
    # transform = transforms.Compose([Rescale(), Normalize()])
    # train_dataset = HeadDataset(cfg.DATASET_DIR, train_annots_path, transform)
    # val_dataset = HeadDataset(cfg.DATASET_DIR, val_annots_path, transform)

    print('[INFO] Load datasets.\n Training set size:{}, Verification set size:{}'
          .format(len(train_dataset), len(val_dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Create HeadDetector instance and Trainer instance
    head_detector = HeadDetector(ratios=cfg.ANCHOR_RATIOS, scales=cfg.ANCHOR_SCALES)
    trainer = Trainer(head_detector).cuda()

    # Load old weight
    # trainer.load(cfg.BEST_MODEL_PATH)
    total_losses=[]
    avg_CoLos=[]
    precisions=[]
    recalls=[]
    print('[INFO] Start training...')
    # cfg.EPOCHS
    for epoch in range(cfg.EPOCHS):
        trainer.reset_meters()
        t1=time.time()
        # file =open("loss.txt",'w')
        for i, data in enumerate(train_dataloader, 1):
            img, boxes, scale = data['img'], data['boxes'], data['scale']
            print("image shape: ",img.shape)
            img = img.cuda().float()
            scale = scale.item()
            # Forward pass and backward pass
            trainer.train_step(img, boxes, scale)

            # Visualize           
            if i % cfg.PLOT_INTERVAL == 0:
              z=trainer.meters['total_loss'].value()
              print('Total loss: ',z[0])
              # print(trainer.rpn_cm.value())
              cm=trainer.rpn_cm.value()
              tp,fp,fn,tn=cm[0,0],cm[0,1],cm[1,0],cm[1,1]
              precision=tp/(tp+fp)
              recall=tp/(tp+fn)
              precisions.append(precision)
              recalls.append(recall)
              total_losses.append(z[0])
            #     trainer.vis.plot_many(trainer.get_meter_data())
            #     origin_img = inverse_normalize(img[0].cpu().numpy())
            #     gt_img = visdom_bbox(origin_img, boxes[0].cpu().numpy())
            #     trainer.vis.img('gt_img', gt_img)
            #     preds, _ = head_detector(img, scale)
            #     pred_img = visdom_bbox(origin_img, preds)
            #     trainer.vis.img('pred_img', pred_img)
            #     trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
            if i%500==0:
                print("Epoch %d: %d/%d images" %(epoch,i,len(train_dataset)))
        avg_CoLos.append(avg_accuracy)
        # file.close()
        t2=time.time()
        # Evaluation
        # avg_accuracy = evaluate(val_dataloader, head_detector)
        # avg_CoLos.append(avg_accuracy)
        # print("[INFO] Epoch {} of {}.".format(epoch + 1, cfg.EPOCHS))
        # print("\tValidate average accuracy: {:.3f}".format(avg_accuracy))
        print("Training time: %d" %(t2-t1))
        # Save current model
        time_str = time.strftime('%m%d%H%M')
        save_path = os.path.join(cfg.MODEL_DIR, 'checkpoint_{}_{:.3f}.pth'.format(time_str, avg_accuracy))
        trainer.save(save_path)

        # Evaluate each 4 epoch
        if epoch %4==0:
          avg_accuracy = evaluate(val_dataloader, head_detector)
          print('Average corect location: ', avg_accuracy)
        # Learning rate decay
        if epoch == 8:
            trainer.scale_lr()
    # avg_accuracy = evaluate(val_dataloader, head_detector)
    # print("\tValidate average accuracy: {:.3f}".format(avg_accuracy))

    # Training loss
    fig = plt.figure()
    plt.plot(2*np.arange(0,len(total_losses)),total_losses,label = "loss")
    plt.xlabel('interations')
    plt.ylabel('loss')
    plt.legend(loc="upper left")
    plt.savefig('loss_BKdata.png')
    # AverAcc
    plt.figure()
    plt.plot(np.arange(0,len(avg_CoLos)),avg_CoLos,label = "Average accuracy")
    plt.xlabel('interations')
    plt.ylabel('Average Accuracy')
    plt.legend(loc="upper left")
    plt.savefig('average_accuracy.png')
    # Precision and recall
    plt.figure()
    plt.plot(recalls,precisions,label="FCHD")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="upper left")
    plt.savefig('recall_precision.png')
def evaluate(val_dataloader, head_detector):
    """
    Given the dataloader of the test split compute the
    average corLoc of the dataset using the head detector 
    model given as the argument to the function. 
    """
    img_counts = 0
    accuracy = 0.0

    for data in val_dataloader:
        img, boxes, scale = data['img'], data['boxes'], data['scale']
        img, boxes = img.cuda().float(), boxes.cuda()
        scale = scale.item()

        preds, _ = head_detector(img, scale)
        gts = boxes[0].cpu().numpy()
        if len(preds) == 0:
            img_counts += 1
        else:
            ious = tools.calc_ious(preds, gts)
            max_ious = ious.max(axis=1)
            correct_counts = len(np.where(max_ious >= 0.5)[0])
            gt_counts = len(gts)
            accuracy += correct_counts / gt_counts
            img_counts += 1

    avg_accuracy = accuracy / img_counts
    return avg_accuracy


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    train()
