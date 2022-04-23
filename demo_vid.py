import torch as th
import numpy as np
import argparse
import os
import cv2
from pathlib import Path
from PIL import Image

from data import transforms, GeneralDataset,draw_image_by_points, PTSconvert2box
from utils import load_configure
from models import obtain_model,remove_module_dict
os.environ['CUDA_VISIBLE_DEVICES']='1'

def face_detect(image, face_detector):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    face = None
    if len(faces) == 0:
        return face

    return [faces[0][0],faces[0][1],faces[0][0]+faces[0][2],faces[0][1]+faces[0][3]]

def paint_lip(image, key_pts):
    paint_img = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)    
    key_pts = key_pts.transpose(1, 0)

    upper_lip_kpt_ids = [49, 50, 51, 52, 53, 54, 55, 65, 64, 63, 62, 61, 49]
    lower_lip_kpt_ids = [49, 60, 59, 58, 57, 56, 55, 65, 66, 67, 68, 61, 49]

    upper_lip_key_pts = []
    for kpt_id in upper_lip_kpt_ids:
        upper_lip_key_pts.append(key_pts[kpt_id - 1])    

    lower_lip_key_pts = []
    for kpt_id in lower_lip_kpt_ids:
        lower_lip_key_pts.append(key_pts[kpt_id - 1])

    upper_lip_key_pts = np.array(upper_lip_key_pts, dtype=np.float32)
    lower_lip_key_pts = np.array(lower_lip_key_pts, dtype=np.float32)
    upper_lip_score = np.mean(upper_lip_key_pts[:, 2])
    lower_lip_score = np.mean(lower_lip_key_pts[:, 2])
    #print(upper_lip_score, lower_lip_score)    
    if upper_lip_score > 0.3 and lower_lip_score > 0.3:
        upper_lip_key_pts = upper_lip_key_pts[:, :2].astype(np.int32)
        lower_lip_key_pts = lower_lip_key_pts[:, :2].astype(np.int32)

        paint_img = cv2.fillPoly(paint_img, [upper_lip_key_pts, lower_lip_key_pts], (0.0, 0.0, 1.0, 0.3))

        mask = paint_img[:, :, 3:]
        paint_img = image * (1.0 - mask) + paint_img[:, :, :3] * mask * 255.0
    else:
        paint_img = image.copy()

    return paint_img

def evaluate(args):
    assert th.cuda.is_available(), 'CUDA is not available'
    th.backends.cudnn.enabled = True
    th.backends.cudnn.benchmark = True

    print('The image is %s' % args.video)
    print('The model is %s' % args.model)

    snapshot = Path(args.model)
    assert snapshot.exists(), 'The model path {:} does not exist'
    snapshot = th.load(str(snapshot))

    # General data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    param = snapshot['args']
    eval_transform = transforms.Compose(
        [transforms.PreCrop(param.pre_crop_expand), transforms.TrainScale2WH((param.crop_width, param.crop_height)),
         transforms.ToTensor(), normalize])

    model_config = load_configure(param.model_config, None)
    dataset = GeneralDataset(eval_transform, param.sigma, model_config.downsample, param.heatmap_type, param.data_indicator)
    dataset.reset(param.num_pts)

    # Build network
    net = obtain_model(model_config, param.num_pts + 1)
    net = net.cuda()
    weights = remove_module_dict(snapshot['state_dict'])
    net.load_state_dict(weights)

    # Face detector
    face_detector = cv2.CascadeClassifier(args.face_detector)

    # Load video
    in_vid = cv2.VideoCapture(args.video)
    in_vid_w, in_vid_h = int(in_vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(in_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_vid_frm_num = int(in_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    in_vid_fourcc = int(in_vid.get(cv2.CAP_PROP_FOURCC))
    in_vid_fps = float(in_vid.get(cv2.CAP_PROP_FPS))
    print('Video dim: %d x %d' % (in_vid_w, in_vid_h))

    # Use temporal?
    temporal = args.temporal
    print("Use temporal:", temporal)
    temp_str = ''
    if temporal:        
        temp_str = '_temporal'

    # Build output video
    out_vid_path = args.video.split('.')[0] + '_' + args.post_str + temp_str + '.mp4' 
    out_vid = cv2.VideoWriter(out_vid_path, in_vid_fourcc, in_vid_fps, (in_vid_w * 2, in_vid_h))

    # Process frame by frame
    for frm_idx in range(in_vid_frm_num):
        success, ori_image = in_vid.read()
        if not success:
            break

        if temporal:
            if frm_idx == 0:
                facebox = face_detect(ori_image, face_detector)
            else:
                facebox = prev_box
        else:
            facebox = face_detect(ori_image, face_detector)

        if facebox is None:
            out_image = np.concatenate((ori_image, ori_image), axis=1)
            print("\n Cannot detect face for frame %d" % frm_idx)
        else:
            #image = cv2.cvtColor()
            [image, _, _, _, _, _, cropped_size], meta = dataset.prepare_img_input(ori_image, facebox)            
            inputs = image.unsqueeze(0).cuda()

            # network forward
            with th.no_grad():
                batch_heatmaps, batch_locs, batch_scos = net(inputs)
            # obtain the locations on the image in the orignial size
            cpu = th.device('cpu')
            np_batch_locs, np_batch_scos, cropped_size = batch_locs.to(cpu).numpy(), batch_scos.to(
                cpu).numpy(), cropped_size.numpy()
            locations, scores = np_batch_locs[0, :-1, :], np.expand_dims(np_batch_scos[0, :-1], -1)

            scale_h, scale_w = cropped_size[0] * 1. / inputs.size(-2), cropped_size[1] * 1. / inputs.size(-1)

            locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], locations[:, 1] * scale_h + \
                                            cropped_size[3]
            prediction = np.concatenate((locations, scores), axis=1).transpose(1, 0)

            prev_pts = np.transpose(locations, (1, 0))            
            prev_box = PTSconvert2box(prev_pts, 0)

            # Draw output
            image = Image.fromarray(ori_image)
            image_kpt = draw_image_by_points(image, prediction, 3, (255, 0, 0), facebox, None,None)
            image_kpt = np.array(image_kpt)
            paint_img = paint_lip(ori_image, prediction)
            out_image = np.concatenate((image_kpt, paint_img), axis=1)
            out_image = out_image.clip(0, 255).astype(np.uint8)            

        out_vid.write(out_image)
        print(" Process frame [%d/%d]" % (frm_idx + 1, in_vid_frm_num), end="\r")
        #break   

        # if frm_idx == 63:
        #     break

    print("")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a video by the trained model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video',            type=str,   help='The evaluation video path.')
    parser.add_argument('--model',            type=str,   help='The snapshot to the saved detector.')
    parser.add_argument('--face_detector',    default='haarcascade_frontalface_alt2.xml',type=str,   help='The face detector')
    parser.add_argument('--post_str',         default='result', type=str, help='The post string to put on the result')    
    parser.add_argument("--temporal",         default=False, action="store_true", help="Use temporal info")
                
    args = parser.parse_args()
    evaluate(args)