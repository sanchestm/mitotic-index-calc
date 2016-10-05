import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from scipy import misc
from skimage import exposure

def texture(img):
    def local(image, value):
        a = np.where(image == value)
        minimum = np.array([(a[0][i]-55)**2 + (a[1][i]-55)**2 for i in range(len(a[0]))])
        minimu_loc = np.where(minimum == minimum.min())
        return (a[0][minimu_loc][0], a[1][minimu_loc][0])
    patch_size = 18
    img_sizes = np.shape(img)
    sorted_img = np.sort(img[int(patch_size/2):img_sizes[0]-int(patch_size/2), int(patch_size/2):img_sizes[1]-int(patch_size/2)].ravel())
    L=len(sorted_img)
    quartiles = [sorted_img[int(L*.25)], sorted_img[int(L*.6)], sorted_img[int(L*.8)], sorted_img[int(L*.95)]]
    loc_quartiles = [local(img, quartiles[0]),local(img, quartiles[1]),local(img, quartiles[2]),local(img, quartiles[3])]
    patches = []
    for i in loc_quartiles:
        patches += [img[i[0]-int(patch_size/2):i[0]+ int(patch_size/2),i[1]-int(patch_size/2):i[1]+ int(patch_size/2)]]
    glcm = []
    for i in patches:
        if i.size == 0: return np.zeros(8)
        glcm += [greycomatrix(i, [5], [0],symmetric=True, normed=True )]
    return [greycoprops(x, 'energy')[0, 0] for x in glcm] + [greycoprops(x, 'correlation')[0, 0] for x in glcm] #+ [greycoprops(x, 'contrast')[0, 0] for x in glcm]


def texture2(image):
    img_size = np.shape(image)[0]
    patch = image[int(img_size/2-img_size/4):int(img_size/2+img_size/4)][int(img_size/2-img_size/4):int(img_size/2+img_size/4)]
    glcm  = greycomatrix(patch,[5],[0], 256)
    return [greycoprops(glcm, 'dissimilarity')[0,0], greycoprops(glcm, 'correlation')[0,0]]
