import csv
import cv2 as cv
import numpy as np
import os

def get_object_info(file_path, result_path, txt_path):

    def find_object(img):
        """Finds the largest object and computes the esd.
        Args:
            img (np.array): gray scale image
        Returns:
            ESD: 2 * sqrt(Area / PI)
        """
        max_val = np.mean(img) - np.std(img)

        img_c = img.copy()
        img_c[np.where(img < max_val)] = 255
        img_c[np.where(img >= max_val)] = 0
        img = img_c
        thresh = img
        # thresh = cv.threshold(img, max_val, 255, cv.THRESH_BINARY)[1]

        cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        cnt_areas = {cv.contourArea(cnt): cnt for cnt in cnts}
        if cnt_areas:
            cnt_areas_list = list(cnt_areas)
            cnt_areas_list.sort()
            max_area = cnt_areas_list[-1]
            esd = 2 * np.sqrt(max_area / np.pi)
            return esd
        else:
            return 0

    fns = sorted(os.listdir(file_path))
    result_path = result_path
    txt_path = txt_path
    info = []
    fileinfo = []

    for fn_r in fns:
        counter = 0
        bounding_info = []
        img_mass = []
        img_esd = []
        txtfn = os.path.join(txt_path, fn_r[:-4])
        fn = os.path.join(file_path, fn_r)
        fn, ending = fn[:-4], fn[-4:]
        
        try:
            pos = fn.find('bar')
            try:
                press = float(fn[pos-5:pos])
                depth = (press-1)/0.1
            except ValueError:
                press = float(fn[pos-4:pos])
                depth = (press-1)/0.1
        except:
            continue
            
        #print(txtfn)
        if ending == ".jpg" or ending == ".tif" or ending == ".png":
            img = cv.imread(fn + ending)
            try:
                with open(txtfn + '.txt') as file:
                    for line in file:
                        d=[]
                        for i in (line.rstrip()).split(' '):
                            d.append(float(i))
                        bounding_info.append(d)
                #print('found: ' + fn +'.txt')
            except:
                continue
            img_h, img_w = img.shape[:2]

            for bbox in bounding_info:
                counter = counter + 1
                instclass, rel_x, rel_y, rel_w, rel_h, conf = bbox
                c_x = round(rel_x * img_w)  #x-position of center
                c_y = round(rel_y * img_h)  #y-position of center
                w = round(rel_w * img_w)
                h = round(rel_h * img_h)
                x = c_x - w // 2            #x-position of top-left corner
                y = c_y - h // 2            #y-position of top-left corner

                crop = img[y:y+h, x:x+w]
                
                gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
                esd = find_object(gray)
                biomass = ((esd / 3)**2.23) * 0.00377 #biomass calculation accourding to Rodriguez & Mullin (1986) 
                img_esd.append(esd)
                img_mass.append(biomass)
                info.append((fn, counter, c_x, c_y, w, h, x, y, esd, instclass, conf, depth, biomass))
            
            sum_esd = 0
            for i in range(counter):
                sum_esd=sum_esd+img_esd[i]
            meanESD = sum_esd/counter

            sum_bio = 0
            for i in range(counter):
                sum_bio=sum_bio+img_mass[i]
            total_biomass = sum_bio

            fileinfo.append([fn_r,fn_r[24:32],fn_r[33:39],fn_r[16:18],depth,counter,meanESD/3,total_biomass])

    with open(result_path + '_detailed.csv', "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Filename", "ID", "Center x in px", "Center y in px", "Width in px", "Height in px", "x in px", "y in px", "ESD in px", "class", "confidence", "depth in meters below surface", "biomass in mg(c) following Rodriguez & Mullin 1986"])
        for row in info:
            writer.writerow(row)    
    with open(result_path + '_overview.csv', "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Filename", "Sampling day", "local time", "Mesocosm-#", "depth in meters below surface", "Copepod-#", "mean-ESD in um", "Biomass in mg(c)"])
        for row in fileinfo:
            writer.writerow(row)   
    

from tkinter.filedialog import askdirectory

IMG_PATH = askdirectory(title="Image Path")
TXT_PATH = askdirectory(title="Label Path")
RESULT_PATH = TXT_PATH + '/results'

print('labels:' + TXT_PATH)
get_object_info(IMG_PATH, RESULT_PATH, TXT_PATH)