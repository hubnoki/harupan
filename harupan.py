
######################################################
# Importing libraries
######################################################
import cv2
import numpy as np
import math
import json

import tkinter as tk
from PIL import Image, ImageOps, ImageTk

import queue
import threading

import time

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
    if len(contours) == 0:
        return contours, img
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
    search_range = 0.25
    return icp_sub(src_pts, dst_pts, max_iter=max_iter, initial_matrix=initial_matrix, search_range=search_range)

# Find optimum affine matrix using ICP algorithm
# src_pts: ndarray, shape is (n_s,2) (n_s: number of points)
# dst_pts: ndarray, shape is (n_d,2) (n_d: number of points, n_d should be larger or equal to n_s)
# initial_matrix: ndarray, shape is (2,3)
# search_range: float number, 0.0 ~ 1.0, the range to search nearest neighbor, 1.0 -> Search in all dst_pts
def icp_sub(src_pts, dst_pts, max_iter=20, initial_matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), search_range=0.5):
    default_affine_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    n_dst = dst_pts.shape[0]
    n_src = src_pts.shape[0]
    if n_dst < n_src:
        # print("icp: Insufficient destination points")
        return default_affine_matrix, False
    if initial_matrix.shape != (2,3):
        print("icp: Illegal shape of initial_matrix")
        return default_affine_matrix, False
    n_search = int(n_dst*search_range)
    M = initial_matrix
    # Store indices of the nearest neighbor point of dst_pts to the converted point of src_pts
    nn_idx = []
    converged = False
    for i in range(max_iter):
        nn_idx_tmp = []
        dst_pts_list = [p for p in dst_pts]
        idx_list = list(range(0,dst_pts.shape[0]))
        first_pt = True
        for p in src_pts:
            # Convert source point with current conversion matrix
            p2 = M @ np.array([p[0], p[1], 1])
            if first_pt:
                # First point should be searched in all destination points
                idx, _ = find_nearest_neighbor(dst_pts_list, p2)
                first_pt = False
            else:
                # Search nearest neighbor point in specified range around the last point
                n = int(min(n_search/2, len(idx_list)/2))
                s = max(len(idx_list) + last_idx - n, 0)
                e = min(len(idx_list) + last_idx + n, 3*len(idx_list))
                pts = (dst_pts_list + dst_pts_list + dst_pts_list)[s:e]
                idx, _ = find_nearest_neighbor(pts, p2)
                # The index acquired above is counted from 's', so actual index must be recovered
                idx = (idx + s) % len(idx_list)
            nn_idx_tmp += [idx_list[idx]]
            last_idx = idx
            del dst_pts_list[idx]
            del idx_list[idx]
        if nn_idx != [] and nn_idx == nn_idx_tmp:
            converged = True
            break
        dst_pts2 = np.zeros_like(src_pts)
        for j,idx in enumerate(nn_idx_tmp):
            dst_pts2[j,:] = dst_pts[idx,:]
        M = estimate_affine_2d(src_pts, dst_pts2)
        nn_idx = nn_idx_tmp
    return M, converged


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
    # print('Number of candidates: ', len(ctrs))
    if len(ctrs) == 0:
        return 0.0, resized_img
    subctrs, _, _ = refine_contours(resized_img, ctrs)
    subctr_datasets = [contour_dataset(ctr) for ctr in subctrs]
    ########
    #### Simple code
    similarities = [get_similarities(d, templates)[0] for d in subctr_datasets]
    #### Code printing progress
    # similarities = []
    # for i,d in enumerate(subctr_datasets):
    #     print(i, end=' ')
    #     similarities += [get_similarities(d, templates)[0]]
    # print('')
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

######################################################
# GUI
######################################################
class harupan_gui(tk.Frame):
    TEXT_CONNECT = 'Connect   '
    TEXT_DISCONNECT = 'Disconnect'
    TEXT_STOP = 'Stop  '
    TEXT_RESUME = 'Resume'

    def __init__(self, master=None, img_queue_size=1, svm_data='harupan_data/harupan_svm_220412.dat', template_data='harupan_data/templates2021.json'):
        super().__init__(master)

        self.cap = cv2.VideoCapture()
        self.open_params = (cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        self.svm = load_svm(svm_data)
        self.templates = load_templates(template_data)

        #### Main window settings ####
        self.master.title('Harupan App')
        self.master.geometry('500x400')
        self.master.protocol('WM_DELETE_WINDOW', self.cleanup_app)
        self.master.bind('<Configure>', self.update_canvas_size)

        #### Sub frames ####
        self.frame_connection = tk.Frame(self)
        self.frame_log = tk.Frame(self.frame_connection, width=120, height=30)
        self.frame_log.propagate(False)
        self.frame_canvas = tk.Frame(self)
        self.frame_canvas.config(relief='ridge', bd=5)
        self.frame_result = tk.Frame(self)

        #### Entries for connection information ####
        self.t_ip = tk.StringVar(value='192.168.1.7')
        self.t_port = tk.StringVar(value='4747')
        self.entry_ip = tk.Entry(self.frame_connection, textvariable=self.t_ip)
        self.entry_port = tk.Entry(self.frame_connection, textvariable=self.t_port)

        #### Connect button ####
        self.t_connect = tk.StringVar(value=self.TEXT_CONNECT)
        self.button_connect = tk.Button(self.frame_connection, textvariable=self.t_connect)
        self.button_connect.bind('<Button-1>', self.event_connect)

        #### Connection log ####
        self.t_log = tk.StringVar()
        self.label_log = tk.Label(self.frame_log, textvariable=self.t_log)

        #### Image canvas ####
        self.canvas_image = tk.Canvas(self.frame_canvas, bg='white')
        self.disp_img = None

        #### Label for calculation result ####
        self.t_calc_result = tk.StringVar(value=' 0 points')
        self.label_points = tk.Label(self.frame_result, bg='black', fg='green', font=('Consolas', 20), textvariable=self.t_calc_result)

        #### Stop button ####
        self.t_stop = tk.StringVar(value=self.TEXT_STOP)
        self.button_stop = tk.Button(self.frame_result, textvariable=self.t_stop)
        self.button_stop.bind('<Button-1>', self.event_stop_button)

        #### Calculation time ####
        self.t_calc_time = tk.StringVar(value='     ms')
        self.label_calc_time = tk.Label(self.frame_result, font=('Consolas', 10), textvariable=self.t_calc_time)

        #### Place widgets ####
        self.pack(expand=True, fill='both')

        self.frame_connection.pack()
        self.frame_canvas.pack(expand=True, fill='both')
        self.frame_result.pack()

        self.entry_ip.grid(row=0, column=0)
        self.entry_port.grid(row=1, column=0)
        self.button_connect.grid(row=0, column=1, rowspan=2, padx=(5,0))
        self.frame_log.grid(row=0, column=2, rowspan=2, padx=(5,0))
        self.label_log.pack(fill='both')

        self.canvas_image.pack(expand=True, fill='both')

        self.label_points.grid(row=0, column=0)
        self.button_stop.grid(row=0, column=1, padx=(5,0))
        self.label_calc_time.grid(row=0, column=2, padx=(5,0))

        self.frame_canvas.update()
        self.w, self.h = self.canvas_image.winfo_width(), self.canvas_image.winfo_height()
        print(f'Canvas size: {self.w},{self.h}')

        #### Start internal threads ####
        self.q_connect = queue.Queue(maxsize=0)
        self.q_img = queue.Queue(maxsize=1)
        self.q_img2 = queue.Queue(maxsize=img_queue_size)
        self.run_flag = True
        self.thread1 = threading.Thread(target=self.update_image, name='thread1')
        self.thread2 = threading.Thread(target=self.cap_process, name='thread2')
        self.thread3 = threading.Thread(target=self.calc_process, name='thread3')
        self.thread1.start()
        self.thread2.start()
        self.thread3.start()
        print(f'Number of threads: {threading.active_count()}')
        for th in threading.enumerate():
            print('  ', th)

    def event_connect(self, e):
        self.t_log.set('')
        if(self.t_connect.get() == self.TEXT_CONNECT):
            url = f'http://{self.t_ip.get()}:{self.t_port.get()}/video'
            self.q_connect.put(url)
            self.t_connect.set(self.TEXT_DISCONNECT)
        else:
            self.q_connect.put(None)
            self.t_connect.set(self.TEXT_CONNECT)

    def update_image(self):
        while self.run_flag:
            val, img = self.q_img2.get()
            if not val:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageOps.pad(img, (self.w,self.h))
            self.disp_img = ImageTk.PhotoImage(image=img)
            self.canvas_image.create_image(self.w/2,self.h/2,image=self.disp_img)

    def _print_log(self, mes):
        print(mes)
        self.t_log.set(mes)

    def cap_process(self):
        while self.run_flag:
            if not self.q_connect.empty():
                url = self.q_connect.get()
                if url == None:
                    self.cap.release()
                    self._print_log('Camera closed')
                elif self.cap.open(url, cv2.CAP_FFMPEG, self.open_params):
                    self._print_log('Camera opened')
                else:
                    self._print_log('Camera open failed')
                    self.t_connect.set(self.TEXT_CONNECT)
            elif self.cap.isOpened():
                ret, img = self.cap.read()
                if not ret:
                    self._print_log('Can\'t receive frame')
                    self.cap.release()
                    self.t_connect.set(self.TEXT_CONNECT)
                elif not self.q_img.full():
                    self.q_img.put((True, img))
    
    def calc_process(self):
        while self.run_flag:
            val, img = self.q_img.get()
            if not val:
                self.q_img2.put((False, None))
                continue
            if self.t_stop.get() == self.TEXT_STOP:
                t = time.time()
                score, img2 = calc_harupan(img, self.templates, self.svm)
                t = time.time() - t
                if not self.q_img2.full():
                    self.t_calc_result.set(f'{score:2.1f} points')
                    self.t_calc_time.set(f'{int(t*1000):4d} ms')
                    self.q_img2.put((True, img2))

    def event_stop_button(self, e):
        s = self.TEXT_RESUME if self.t_stop.get() == self.TEXT_STOP else self.TEXT_STOP
        self.t_stop.set(s)

    def update_canvas_size(self, e):
        self.w, self.h = self.canvas_image.winfo_width(), self.canvas_image.winfo_height()

    def cleanup_app(self):
        self.run_flag = False

        # Put dummy data to finish thread1(update_image()), thread3(calc_process())
        if self.q_img.empty():
            self.q_img.put((False, None))
        if self.q_img2.empty():
            self.q_img2.put((False, None))

        self.thread1.join(timeout=10)
        self.thread2.join(timeout=10)
        self.thread3.join(timeout=10)

        print(f'Number of threads: {threading.active_count()}')
        for th in threading.enumerate():
            print('  ', th)

        if self.cap.isOpened():
            self.cap.release()

        self.master.destroy()

######################################################
# main
######################################################
def main():
    root = tk.Tk()
    svm_data='harupan_svm_220412.dat'
    template_data='templates2021.json'
    app = harupan_gui(master=root, img_queue_size=1, svm_data=svm_data, template_data=template_data)
    app.mainloop()


if __name__ == '__main__':
    main()
