from SimpleCV import  HaarLikeFeatureExtractor, MorphologyFeatureExtractor, HueHistogramFeatureExtractor
from SimpleCV import Image
import numpy as np
import sys

def getExtractors(image):
    image = Image(image)
    hhfe = HueHistogramFeatureExtractor(10).extract(image)
    MFE = MorphologyFeatureExtractor().extract(image)
    haarfe = HaarLikeFeatureExtractor(fname='/home/thiago/SimpleCV/SimpleCV/Features/haar.txt').extract(image)
    print(hhfe+ haarfe + MFE)
    return [hhfe+ haarfe + MFE]
