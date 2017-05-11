from numpy import array
import glob
from numpy import asarray


def get_all_image_names(path):
    filetypes = [".png", ".jpg", "jpeg", "tif", "tiff"]
    imlist = []
    for filetype in filetypes:
        imlist += glob.glob(path + "/*"+ filetype)
    return imlist

def CreateIndividualCellPatches(row):
    result = []
    for blob in row["blobs"]:
        y,x,r = blob
        y, x = int(y), int(x)
        result += [row["photo"][y:y+100,x:x+100]]
    return result

def checksizephoto(photo):
    if photo.shape[0] > 101 and photo.shape[1] > 101: return True
    else: return False
