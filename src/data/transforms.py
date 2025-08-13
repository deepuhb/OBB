import numpy as np
import torch
import cv2

def resize_longest_edge(img, size):
    h,w = img.shape[:2]
    scale = size / max(h,w)
    nh, nw = int(round(h*scale)), int(round(w*scale))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h = size - nh
    pad_w = size - nw
    top = pad_h//2; bottom = pad_h-top
    left= pad_w//2; right = pad_w-left
    im = cv2.copyMakeBorder(im, top,bottom,left,right, cv2.BORDER_CONSTANT, value=(114,114,114))
    return im, scale, (left, top)

def transform_sample(sample, img_size):
    img = sample['image']
    boxes = sample['boxes']
    quads = sample['quads']
    kpts  = sample['kpts']
    angles= sample['angles']
    labels= sample['labels']

    im, scale, (dx,dy) = resize_longest_edge(img, img_size)
    if boxes.shape[0]>0:
        boxes = boxes*scale
        boxes[:,[0,2]] += dx
        boxes[:,[1,3]] += dy
    if quads.shape[0]>0:
        quads = quads*scale
        quads[:,:,0] += dx
        quads[:,:,1] += dy
    if kpts.shape[0]>0:
        kpts = kpts*scale
        kpts[:,0] += dx
        kpts[:,1] += dy

    im = im.astype(np.float32)/255.0
    im = (im - np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
    im = torch.from_numpy(im.transpose(2,0,1)).float()

    import numpy as _np
    return {
        'image': im,
        'boxes': torch.from_numpy(boxes).float() if isinstance(boxes, _np.ndarray) else boxes,
        'quads': torch.from_numpy(quads).float() if isinstance(quads, _np.ndarray) else quads,
        'kpts':  torch.from_numpy(kpts).float()  if isinstance(kpts, _np.ndarray) else kpts,
        'angles': torch.from_numpy(angles).float() if isinstance(angles, _np.ndarray) else angles,
        'labels': torch.from_numpy(labels).long() if isinstance(labels, _np.ndarray) else labels,
        'orig_size': torch.tensor([img.shape[0], img.shape[1]])
    }
