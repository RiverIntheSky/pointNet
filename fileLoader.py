from utils.pc_util import *
import numpy as np
import random
import os
from PIL import Image
import time

dic = {'Rigid_0'      : 0,
       'Rigid_3'      : 1,
       'Rigid_4'      : 2,
       'Rigid_6'      : 3,
       'tower_A'      : 4,
       'tower_B'      : 5,
       'tower_C'      : 6,
       'wall'         : 7,
       'Bridge'       : 8,
       'Bridge_Stairs': 9,
       'Module_C'     :10,
       'Roof'         :11,
       'Stairs'       :12,
       'module_A'     :13,
       'Bridge_Large' :14,
       'Bridge_Small' :15,
       'Building_Big' :16,
       'Building_Small':17}
	
def MTL(filename):
    contents = {}
    mtl = None
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise ValueError, "mtl file doesn't start with newmtl stmt"
        else:
            mtl[values[0]] = map(float, values[1:])
    return contents

class OBJ:
    def __init__(self, filename):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.faces = []
        self.lines = []

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = map(float, values[1:4])
                self.vertices.append(v)
            elif values[0] in ('usemtl', 'usemat'):
                material = dic[values[1]]
            elif values[0] == 'mtllib':
                mtl_file = os.path.join(os.path.dirname(filename), values[1])
                self.mtl = MTL(mtl_file)
            elif values[0] == 'f':
                face = []
                for v in values[1:]:
                    w = v.split('/')
                    idx = int(w[0])
                    if idx > 0:
                        idx -= 1
                    face.append(idx)
                    if len(self.vertices[idx]) == 3:
                        self.vertices[idx].append(material)
                self.faces.append((face, material))   
            elif values[0] == 'l':
                line = []
                for v in values[1:]:
                    idx = int(v)
                    if idx > 0:
                        idx -= 1
                    line.append(idx)
                    if len(self.vertices[idx]) == 3:
                        self.vertices[idx].append(material)
                self.lines.append((line, material))

def random_point_on_triangle(p1, p2, p3):
    r = np.random.random(2)
    return (1 - np.sqrt(r[0])) * p1 + (np.sqrt(r[0]) * (1 - r[1])) * p2 + (np.sqrt(r[0]) * r[1]) * p3
	
def random_point_on_line(p1, p2):
    r = np.random.random()
    return p1 * (1 - r) + p2 * r

def sample_points(obj, n_samples):
    # sample points and fir into unit sphere
    points_seg = obj.vertices
    random.shuffle(points_seg)           # maybe will be shuffled in the network input??
	
    # downsample
    if np.shape(points_seg)[0] > n_samples:
        points_seg = random.sample(points_seg, n_samples)
		
    # divide points and segment
    points_seg = np.array(points_seg)
    points = points_seg[:,:3].astype(float)
    seg = points_seg[:,-1]
	
    # upsample on random face/line
    if np.shape(points_seg)[0] < n_samples:
        samples_left = n_samples - np.shape(points_seg)[0]
        fc = obj.faces
        ln = obj.lines
        fc_quo = len(fc) / (len(fc) + len(ln))
        while samples_left > 0:
            r = np.random.random()
            if r < fc_quo:
                # sample on face
                f = random.sample(fc, 1)[0]
                fv = f[0]
                mat = f[1]
                nv = random_point_on_triangle(points[fv[0]], points[fv[1]], points[fv[2]])
            else:
                #sample on line
                l = random.sample(ln, 1)[0]
                lv = l[0]
                mat = l[1]
                nv = random_point_on_line(points[lv[0]], points[lv[1]])
            points = np.append(points, np.array([nv]), axis = 0)
            seg = np.append(seg, mat)
            samples_left -= 1


    center = np.mean(points, axis=0, keepdims=True)
    scale = np.max(np.amax(points, axis=0) - np.amin(points, axis=0))
    points = (points - center) / scale

    return points, seg
	
def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]
    
def loadDataFile_with_seg(filename, n_samples):
    obj = OBJ(filename)
    points, seg = sample_points(obj, n_samples)
    category = filename.split('/')[-2]
    # category = category.replace('test_', '')
    #############################
    if category == 'church':
        label = 0
    elif category == 'sand_castle':
        label = 1
    elif category == 'playground':
        label = 2
    elif category == 'moon_base':
        label = 3
    else:
        pass
    #############################
    return np.expand_dims(points, 0), np.array([label]), np.expand_dims(seg, 0)

#n_samples = 1024				
#obj = OBJ("c28.obj")     
#points, seg = sample_points(obj, n_samples)			                           
#im_array = point_cloud_three_views(points)
#img = Image.fromarray(np.uint8(im_array*255.0))
#img.save('church.jpg')
