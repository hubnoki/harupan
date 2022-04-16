
######################################################
# Importing libraries
######################################################
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import copy
import random
import json

######################################################
# Detecting contours
######################################################
def reduce_resolution(img, res_th=800):
    h, w, chs = img.shape
    if h > res_th or w > res_th:
        k = float(res_th)/h if w > h else float(res_th)/w
    else:
        k = 1.0
    rtn_img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return rtn_img

def harupan_binarize(img, sat_th=100):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Convert hue value (rotation, mask by saturation)
    hsv[:,:,0] = np.where(hsv[:,:,0] < 50, hsv[:,:,0]+180, hsv[:,:,0])
    hsv[:,:,0] = np.where(hsv[:,:,1] < sat_th, 0, hsv[:,:,0])
    # Thresholding with cv2.inRange()
    binary_img = cv2.inRange(hsv[:,:,0], 135, 190)
    return binary_img

def detect_candidate_contours(image, res_th=800, sat_th=100):
    img = reduce_resolution(image, res_th)
    binimg = harupan_binarize(img, sat_th)
    # Retrieve all points on the contours (cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(binimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Pick up contours that have no parents
    indices = [i for i,hier in enumerate(hierarchy[0,:,:]) if hier[3] == -1]
    # Pick up contours that reside in above contours
    indices = [i for i,hier in enumerate(hierarchy[0,:,:]) if (hier[3] in indices) and (hier[2] == -1) ]
    contours = [contours[i] for i in indices]
    contours = [ctr for ctr in contours if cv2.contourArea(ctr) > float(res_th)*float(res_th)/4000]
    return contours, img

# image: Entire image containing multiple contours
# contours: Contours contained in "image" (Retrieved by cv2.findContours(), the origin is same as "image")
def refine_contours(image, contours):
    subctrs = []
    subimgs = []
    binimgs = []
    thresholds = []
    n_ctrs = []
    for ctr in contours:
        img, _ = create_contour_area_image(image, ctr)
        # Thresholding using G value in BGR format
        thresh, binimg = cv2.threshold(img[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Add black region around thresholded image, to detect contours correctly
        binimg = cv2.copyMakeBorder(binimg, 2,2,2,2, cv2.BORDER_CONSTANT, 0)
        ctrs2, _ = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_len = 0
        for ctr2 in ctrs2:
            if max_len <= ctr2.shape[0]:
                max_ctr = ctr2
                max_len = ctr2.shape[0]
        subctrs += [max_ctr]
        subimgs += [img]
        binimgs += [binimg]
        thresholds += [thresh]
        n_ctrs += [len(ctrs2)]
    debug_info = (binimgs, thresholds, n_ctrs)
    return subctrs, subimgs, debug_info

######################################################
# Auxiliary functions
######################################################
def create_contour_area_image(img, ctr):
    x,y,w,h = cv2.boundingRect(ctr)
    rtn_img = img[y:y+h,x:x+w,:].copy()
    rtn_ctr = ctr.copy()
    origin = np.array([x,y])
    for c in rtn_ctr:
        c[0,:] -= origin
    return rtn_img, rtn_ctr

# ctr: Should be output of create_contour_area_image() (Origin of points is the origin of bounding box)
# img_shape: Optional, tuple of (image_height, image_width), if omitted, calculated from ctr
def create_solid_contour(ctr, img_shape=(int(0),int(0))):
    if img_shape == (int(0),int(0)):
        _,_,w,h = cv2.boundingRect(ctr)
    else:
        h,w = img_shape
    img = np.zeros((h,w), 'uint8')
    img = cv2.drawContours(img, [ctr], -1, 255, -1)
    return img

# ctr: Should be output of create_contour_area_image() (Origin of points is the origin of bounding box)
def create_upright_solid_contour(ctr):
    ctr2 = ctr.copy()
    (cx,cy),(w,h),angle = cv2.minAreaRect(ctr2)
    M = cv2.getRotationMatrix2D((cx,cy), angle, 1)
    for i in range(ctr2.shape[0]):
        ctr2[i,0,:] = ( M @ np.array([ctr2[i,0,0], ctr2[i,0,1], 1]) ).astype('int')
    rect = cv2.boundingRect(ctr2)
    img = np.zeros((rect[3],rect[2]), 'uint8')
    ctr2 -= rect[0:2]
    M[:,2] -= rect[0:2]
    img = cv2.drawContours(img, [ctr2], -1, 255,-1)
    return img, M, ctr2


######################################################
# Dataset classes
######################################################
class contour_dataset:
    def __init__(self, ctr):
        self.ctr = ctr.copy()
        self.rrect = cv2.minAreaRect(ctr)
        self.box = cv2.boxPoints(self.rrect)
        self.solid = create_solid_contour(ctr)
        n = 100
        if n >= ctr.shape[0]:
            self.pts = np.array([p for p in ctr[:,0,:]])
        else:            
            r = n / ctr.shape[0]
            self.pts = np.zeros((100,2), 'int')
            pts = []
            for i in range(ctr.shape[0]):
                f = math.modf(i*r)[0] 
                if (f <= r/2) or (f > 1.0 - r/2):
                    pts += [ctr[i,0,:]]
            self.pts = np.array(pts)

class template_dataset:
    def __init__(self, ctr, num, selected_idx=[0]):
        self.ctr = ctr.copy()
        self.num = num
        self.rrect = cv2.minAreaRect(ctr)
        self.box = cv2.boxPoints(self.rrect)
        if num == 0:
            self.solid,_,_ = create_upright_solid_contour(ctr)
        else:
            self.solid = create_solid_contour(ctr)
        self.pts = np.array([ctr[idx,0,:] for idx in selected_idx])


######################################################
# ICP
######################################################
# pts: list of 2D points, or ndarray of shape (n,2)
# query: 2D point to find nearest neighbor
def find_nearest_neighbor(pts, query):
    min_distance_sq = float('inf')
    min_idx = 0
    for i, p in enumerate(pts):
        d = np.dot(query - p, query - p)
        if(d < min_distance_sq):
            min_distance_sq = d
            min_idx = i
    return min_idx, np.sqrt(min_distance_sq)

# src, dst: ndarray, shape is (n,2) (n: number of points)
def estimate_affine_2d(src, dst):
    n = min(src.shape[0], dst.shape[0])
    x = dst[0:n].flatten()
    A = np.zeros((2*n,6))
    for i in range(n):
        A[i*2,0] = src[i,0]
        A[i*2,1] = src[i,1]
        A[i*2,2] = 1
        A[i*2+1,3] = src[i,0]
        A[i*2+1,4] = src[i,1]
        A[i*2+1,5] = 1
    M = np.linalg.inv(A.T @ A) @ A.T @ x
    return M.reshape([2,3])

# Find optimum affine matrix using ICP algorithm
# src_pts: ndarray, shape is (n_s,2) (n_s: number of points)
# dst_pts: ndarray, shape is (n_d,2) (n_d: number of points, n_d should be larger or equal to n_s)
# initial_matrix: ndarray, shape is (2,3)
def icp(src_pts, dst_pts, max_iter=20, initial_matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])):
    default_affine_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if dst_pts.shape[0] < src_pts.shape[0]:
        print("icp: Insufficient destination points")
        return default_affine_matrix, False
    if initial_matrix.shape != (2,3):
        print("icp: Illegal shape of initial_matrix")
        return default_affine_matrix, False
    M = initial_matrix
    # Store indices of the nearest neighbor point of dst_pts to the converted point of src_pts
    nn_idx = []
    for i in range(max_iter):
        nn_idx_tmp = []
        dst_pts_list = [p for p in dst_pts]
        idx_list = list(range(0,dst_pts.shape[0]))
        for p in src_pts:
            p2 = M @ np.array([p[0], p[1], 1])
            idx, d = find_nearest_neighbor(dst_pts_list, p2)
            nn_idx_tmp += [idx_list[idx]]
            del dst_pts_list[idx]
            del idx_list[idx]
        if nn_idx != [] and nn_idx == nn_idx_tmp:
            break
        dst_pts2 = np.zeros_like(src_pts)
        for j,idx in enumerate(nn_idx_tmp):
            dst_pts2[j,:] = dst_pts[idx,:]
        M = estimate_affine_2d(src_pts, dst_pts2)
        nn_idx = nn_idx_tmp
        if i == max_iter -1:
            return M, False
    return M, True


######################################################
# Calculating similarity and determining the number
######################################################
def binary_image_similarity(img1, img2):
    if img1.shape != img2.shape:
        print('binary_image_similarity: Different image size')
        return 0.0
    xor_img = cv2.bitwise_xor(img1, img2)
    return 1.0 - np.float(np.count_nonzero(xor_img)) / (img1.shape[0]*img2.shape[1])

# src, dst: contour_dataset or template_dataset (holding member variables box, solid)
def get_transform_by_rotated_rectangle(src, dst):
    # Rotated patterns are created when starting index is slided
    dst_box2 = np.vstack([dst.box, dst.box])
    max_similarity = 0.0
    max_converted_img = np.zeros((dst.solid.shape[1], dst.solid.shape[0]), 'uint8')
    for i in range(4):
        M = cv2.getAffineTransform(src.box[0:3], dst_box2[i:i+3])
        converted_img = cv2.warpAffine(src.solid, M, dsize=(dst.solid.shape[1], dst.solid.shape[0]), flags=cv2.INTER_NEAREST)
        similarity = binary_image_similarity(converted_img, dst.solid)
        if similarity > max_similarity:
            M_rtn = M
            max_similarity = similarity
            max_converted_img = converted_img
    return M_rtn, max_similarity, max_converted_img

def get_similarity_with_template(target_data, template_data, sim_th_high=0.95, sim_th_low=0.7):
    _,(w1,h1), _ = target_data.rrect
    _,(w2,h2), _ = template_data.rrect
    r = w1/h1 if w1 < h1 else h1/w1
    r = r * h2/w2 if w2 < h2 else r * w2/h2
    M, sim_init, _ = get_transform_by_rotated_rectangle(template_data, target_data)
    if sim_init > sim_th_high or sim_init < sim_th_low or r > 1.4 or r < 0.7:
        dsize = (template_data.solid.shape[1], template_data.solid.shape[0])
        flags = cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP
        converted_img = cv2.warpAffine(target_data.solid, M, dsize=dsize, flags=flags)
        return sim_init, converted_img
    M, _ = icp(template_data.pts, target_data.pts, initial_matrix=M)
    Minv = cv2.invertAffineTransform(M)
    converted_ctr = np.zeros_like(target_data.ctr)
    for i in range(target_data.ctr.shape[0]):
        converted_ctr[i,0,:] = (Minv[:,0:2] @ target_data.ctr[i,0,:]) + Minv[:,2]
    converted_img = create_solid_contour(converted_ctr, img_shape=template_data.solid.shape)
    val = binary_image_similarity(converted_img, template_data.solid)
    return val, converted_img

def get_similarity_with_template_zero(target_data, template_data):
    dsize = (template_data.solid.shape[1], template_data.solid.shape[0])
    converted_img = cv2.resize(target_data.solid, dsize=dsize, interpolation=cv2.INTER_NEAREST)
    val = binary_image_similarity(converted_img, template_data.solid)
    return val, converted_img

def get_similarities(target, templates):
    similarities = []
    converted_imgs = []
    for tmpl in templates:
        if tmpl.num == 0:
            sim,converted_img = get_similarity_with_template_zero(target, tmpl)
        else:
            sim,converted_img = get_similarity_with_template(target, tmpl)
        similarities += [sim]
        converted_imgs += [converted_img]
    return similarities, converted_imgs

def calc_harupan(img, templates, svm):
    ctrs, resized_img = detect_candidate_contours(img, sat_th=50)
    print('Number of candidates: ', len(ctrs))
    subctrs, _, _ = refine_contours(resized_img, ctrs)
    subctr_datasets = [contour_dataset(ctr) for ctr in subctrs]
    ########
    #### Simple code
    # similarities = [get_similarities(d, templates)[0] for d in subctr_datasets]
    #### Code printing progress
    similarities = []
    for i,d in enumerate(subctr_datasets):
        print(i, end=' ')
        similarities += [get_similarities(d, templates)[0]]
    print('')
    ########
    _, result = svm.predict(np.array(similarities, 'float32'))
    result = result.astype('int')
    score = 0.0
    texts = {0:'0', 1:'1', 2:'2', 3:'3', 5:'.5'}
    font = cv2.FONT_HERSHEY_SIMPLEX
    for res, ctr in zip(result, ctrs):
        if res[0] == 5:
            score += 0.5
        elif res[0] != -1:
            score += res[0]
        
        # Annotating recognized numbers for confirmation
        if res[0] != -1:
            resized_img = cv2.drawContours(resized_img, [ctr], -1, (0,255,0), 3)
            x,y,_,_ = cv2.boundingRect(ctr)
            resized_img = cv2.putText(resized_img, texts[res[0]], (x,y), font, 1, (230,230,0), 5)
    return score, resized_img

######################################################
# Loading template data and SVM model
######################################################
def load_svm(filename):
    return cv2.ml.SVM_load(filename)

def load_templates(filename):
    with open(filename, mode='r') as f:
        load_data = json.load(f)
        templates_rtn = []
        for d in load_data:
            templates_rtn += [template_dataset(np.array(d['ctr']), d['num'], d['pts'])]
    return templates_rtn

