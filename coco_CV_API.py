import numpy as np
import copy
import cv2

def anchor_gen(size_r,size_c,rpn_stride,anchor_scales,anchor_rations):
    '''
    :param size_x: feature rows INT
    :param size_y: feature cols INT
    :param rpn_stride: the proportion of original image and feature INT
    :param anchor_scales: the scales of anchor in original image [...]
    :param anchor_rations: the ration in one anchor [...]
    :return: some boxes info include 4 coordinates
    '''
    scales,rations=np.meshgrid(anchor_scales,anchor_rations)
    scales,rations=scales.flatten(),rations.flatten()
    scalesx=scales*np.sqrt(rations)
    scalesy=scales/np.sqrt(rations)
    shift_x=np.arange(0,size_r+1)*rpn_stride
    shift_y=np.arange(0,size_c+1)*rpn_stride
    shift_x,shift_y=np.meshgrid(shift_x,shift_y)
    centerx,anchorx=np.meshgrid(shift_x,scalesx)
    centery,anchory=np.meshgrid(shift_y,scalesy)
    anchor_center=np.stack([centery,centerx],axis=2).reshape(-1,2)
    anchor_size=np.stack([anchory,anchorx],axis=2).reshape(-1,2)
    boxes=np.concatenate([anchor_center-0.5*anchor_size,anchor_center+0.5*anchor_size],axis=1)
    return boxes

def filter_boxes(image_row,image_col,boxes):
    ls_boxes=boxes.tolist()
    new_boxes=[element for element in ls_boxes if not(element[0]<0 or element[1]<0 or element[2]>image_col or element[3]>image_row)]
    return new_boxes

def draw_boxes(image,boxes,classes=None,coco=None,resize=False):
    '''
    :param image: image cv_form
    :param boxes: a list storage some boxes
    :param classes: classes list
    :param coco: coco object
    :return: image
    '''
    image=copy.deepcopy(image)
    if classes!=None and resize==False:
        assert len(boxes)==len(classes) and coco!=None,'boxes can not match classes OR coco is requested but None'
        for bbox,c in zip(boxes,classes):
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), ((int(bbox[0]) + int(bbox[2])), (int(bbox[1]) + int(bbox[3]))), (0, 255, 0), 1)
            info=coco.loadCats(c)
            txt=info[0]['name']
            cv2.putText(image, txt, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 255), 1)
    elif coco==None:
        for bbox in boxes:
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)
    elif classes!=None and resize==True:
        for bbox,c in zip(boxes,classes):
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
            info=coco.loadCats(c)
            txt=info[0]['name']
            cv2.putText(image, txt, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 255), 1)
    else:
        raise ('param wrone!!!!!!!!!!')
    return image

def union(anchor_a,anchor_b,area_overlap):
    area_a=(anchor_a[2]-anchor_a[0])*(anchor_a[3]-anchor_a[1])
    area_b=(anchor_b[2]-anchor_b[0])*(anchor_b[3]-anchor_b[1])
    area_union=area_a+area_b-area_overlap
    return area_union

def overlap(box_a,box_b):
    x=max(box_a[0],box_b[0])
    y=max(box_a[1],box_b[1])
    w=min(box_a[2],box_b[2])-x
    h=min(box_a[3],box_b[3])-y
    if w<0 or h<0:
        return 0
    return w*h

def iou(box_a,box_b):
    '''
    :param box_a: get one box what's form is [x1,y1,x2,y2]
    :param box_b: is like to box_a
    :return: iou
    '''
    if box_a[0]>=box_a[2] or box_a[1]>=box_a[3] or box_b[0]>=box_b[2] or box_b[1]>=box_b[3]:
        return 0.0
    area_overleap=overlap(box_a,box_b)
    area_union=union(box_a,box_b)
    iou_score=float(area_overleap)/float(area_union+1e-6)
    return iou_score

def normalizate_image(cocoimg,cocoann,coco,minside=600):
    r,c=cocoimg.shape[0],cocoimg.shape[1]
    new_img=copy.deepcopy(cocoimg)
    masks,mask=getmask(cocoimg,cocoann,coco)
    classes=getclasses(cocoann)
    # print(classes)
    new_ann = {}
    if r<=c:
        f=float(minside)/r
        resize_c=int(f*c)
        resize_r=minside
    else:
        f=float(minside)/c
        resize_r=int(f*r)
        resize_c=minside
    new_img=cv2.resize(new_img,(resize_c,resize_r),interpolation=cv2.INTER_CUBIC)
    masks=[cv2.resize(m,(resize_c,resize_r),interpolation=cv2.INTER_CUBIC) for m in masks]
    mask=cv2.resize(mask,(resize_c,resize_r),interpolation=cv2.INTER_CUBIC)
    new_boxes=[]
    for stuff in cocoann:
        box=stuff['bbox']
        new_box=[int(box[0]*(resize_r/r)),int(box[1]*(resize_c/c)),int((box[0]+box[2])*(resize_r/r)),int((box[1]+box[3])*(resize_c/c))]
        new_boxes.append(new_box)
    new_ann['boxes']=new_boxes
    new_ann['masks']=masks
    new_ann['mask']=mask
    new_ann['classes']=classes
    return new_img,new_ann


def getmask(cocoimg,cocoann,coco):
    '''
    :param cocoimg: image cv_form
    :param cocoann: coco ann
    :param coco: coco object
    :return: 1.a list of ever one mask of instance of image. 2.a label image that total masks is integrated in the label image
    '''
    masks=[]
    row,col=cocoimg.shape[0],cocoimg.shape[1]
    mask=np.zeros((row,col),dtype=np.float32)
    for ann in cocoann:
        m=coco.annToMask(ann)   #consists of 0 and 1
        masks.append(m)
        cat_id=ann['category_id']
        m=m.astype(np.float32) * cat_id
        mask[m>0]=m[m>0]
    masks=np.asarray(masks)
    masks=masks.astype(np.uint8)
    mask=mask.astype(np.uint8)
    return masks,mask

def getclasses(cocoann):
    classes=[]
    for ann in cocoann:
        classes.append(ann['category_id'])
    return classes

def draw_mask(image,masks,alpha=0.5):
    image=copy.deepcopy(image)
    def random_colors(N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        from:https://blog.csdn.net/fanzonghao/article/details/88111159
        """
        import random
        import colorsys
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors
    colors=random_colors(len(masks))
    for i in range(len(masks)):
        mask = masks[i]
        color = colors[i]
        for c in range(3):
            image[:, :, c] = np.where(mask ==1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
    return image

def coco_visualization(image,cocoann,coco,resize=False):
    '''
    :param image: image cv form
    :param cocoann: coco ann or normalizate_image's new_ann
    :param coco: coco object
    :param resize: tow models, if resize is False, cocoann mean coco ann, otherwise mean normalizate_image's new_ann
    :return: a cv form image where draw boxes an masks
    '''
    if resize:
        boxes=cocoann['boxes']
        masks=cocoann['masks']
        mask=cocoann['mask']
        classes=cocoann['classes']
        image=draw_boxes(image,boxes,classes,coco,resize=True)
        image = draw_mask(image, masks)
    else:
        classes=getclasses(cocoann)
        masks,mask=getmask(image,cocoann,coco)
        boxs=[bbox['bbox'] for bbox in cocoann]
        image=draw_boxes(image,boxs,classes,coco)
        image=draw_mask(image,masks)
    return image
