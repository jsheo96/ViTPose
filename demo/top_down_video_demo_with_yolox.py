# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo
import sys
sys.path.insert(0, '/Data/ByteTrack')
from yolox.exp import get_exp
import torch
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from yolox.data.data_augment import preproc
from yolox.utils import fuse_model, postprocess

def get_speed(length, online_target, prev_center):
    t,l,w,h = online_target.tlwh
    c1 = [t + w/2, l + h/2]
    vector = np.array(c1) - np.array(prev_center)
    return np.linalg.norm(vector)

def get_area(keypoints):
    #17 3
    outline_order = [1,2,6,7,8,9,10,11,5,4]
    cnt = np.array([keypoints[i][:2] for i in outline_order])
    area = cv2.contourArea(cnt)
    return area

def get_length(keypoints):
    head = keypoints[0][:2]
    tail = keypoints[9][:2]
    total_length_vector = tail - head
    total_length = np.linalg.norm(total_length_vector)
    return total_length

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
def vis_information(img, pose_results, online_targets, prev_center_from_id):
    lengths = []
    for i, pose_result in enumerate(pose_results):
        print(pose_result)
        bbox = pose_result['bbox']
        keypoints = pose_result['keypoints']
        x1,y1 = int(bbox[0]), int(bbox[1])
        id_ = online_targets[i].track_id
        org = (x1, y1-3)
        area = get_area(keypoints)
        length = get_length(keypoints)
        if prev_center_from_id == {}:
            speed = 0
        else:
            if online_targets[i].track_id not in prev_center_from_id.keys():
                speed = 0
            else:
                online_target = online_targets[i]
                prev_center = prev_center_from_id[online_target.track_id] # Horrible code
                speed = get_speed(length, online_target, prev_center)
                lengths.append(length)
        img = cv2.putText(img, 'ID: {}, Area: {:.0f}, Length: {:.0f}, Speed: {:.2f}'.format(id_, area, length, speed), org, 1,1.5,color=(203,192,255), thickness=2)
    fig = plt.figure()
    fig.add_subplot(111)
    plt.rcParams['axes.facecolor']='black'
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.hist(lengths, range=(0,450), bins=10, rwidth=0.5,color='white')
    plt.xlabel('Length')
    plt.ylim((0,12))
    plt.xlim((0,450))
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(width)
    height = int(height)
    hist_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    hist_img = cv2.resize(hist_img, None, fx=0.75, fy=0.75)
    hist_img = hist_img[int(hist_img.shape[0]*0.13):int(-hist_img.shape[0]*0.02), int(hist_img.shape[1]*0.09):int(-hist_img.shape[1]*0.1), :]
    img[img.shape[0] - hist_img.shape[0]:, :hist_img.shape[1]] = hist_img
    # cv2.imshow("", img)
    # cv2.waitKey()
    return img

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False,
        confthre=0.5
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = confthre
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('exp_file', help='Exp file for ByteTrack')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument('--thickness',type=int,default=1,help='Link thickness for visualization')
    parser.add_argument('--bbox_color',default='blue',help='Link thickness for visualization')
    parser.add_argument('--bbox_thickness',type=int,default=1,help='Link thickness for visualization')
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--mot20', default=True, action='store_false')
    parser.add_argument('--no_bbox', default=False, action='store_true')

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    # build the pose model from a config file and a checkpoint file

    exp = get_exp(args.exp_file, None)
    det_model = exp.get_model().to(args.device)
    det_model.eval()
    ckpt = torch.load(args.det_checkpoint, map_location="cpu")
    det_model.load_state_dict(ckpt["model"])
    det_model = fuse_model(det_model)
    det_model = det_model.half()  # to FP16
    predictor = Predictor(det_model, exp, None, None, args.device, True)
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()

    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f'Faild to load video file {args.video_path}'

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    prev_center_from_id = {}
    n = 0
    online_targets = []
    while (cap.isOpened()):
        n += 1
        #if n > 150:
        #    break
        flag, img = cap.read()
        if not flag:
            break

        # keep the person class bounding boxes.
        outputs = predictor.inference(img, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], img.shape[:2], predictor.test_size)
        
        bboxes = [t.tlwh for t in online_targets] # t.tlwh gives in (x1, y1, w, h) order fuck
        bboxes = [[t, l, t+w, l+h] for t,l,w,h in bboxes]
        person_results = [{'bbox': np.array([x1,y1,x2,y2])} for x1,y1,x2,y2 in bboxes]
        # person_results = [{'bbox': np.array([0, 0, size[0], size[1]])}]

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # show the results
        if args.no_bbox:
            for pose_result in pose_results:
                del pose_result['bbox']
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False,
            bbox_color=args.bbox_color,
            bbox_thickness=args.bbox_thickness)
        # vis_img = vis_information(vis_img, pose_results, online_targets, prev_center_from_id)
        
        timer.toc()
        cv2.putText(vis_img, 'FPS: {:.2f} Number of Fish: {}'.format(1./ max(1e-5, timer.average_time), len(online_targets)), (0,20), 1, 1.5, (0,0,255), thickness=2)
        if args.show:
            cv2.imshow('Image', vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break
        prev_center_from_id = {t.track_id:[t.tlwh[0]+t.tlwh[2]/2,t.tlwh[1]+t.tlwh[3]/2] for t in online_targets}


    cap.release()
    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
