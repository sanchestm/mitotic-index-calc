#!/usr/bin/python2
from SimpleCV import HueHistogramFeatureExtractor, HaarLikeFeatureExtractor, MorphologyFeatureExtractor
from SimpleCV import Image
import sys

def getExtractors(image_path):
    image = Image(image_path)
    hhfe = HueHistogramFeatureExtractor(10).extract(image)
    #ehfe = EdgeHistogramFeatureExtractor().extract(image)
    MFE = MorphologyFeatureExtractor().extract(image)
    haarfe = HaarLikeFeatureExtractor(fname='/home/thiago/SimpleCV/SimpleCV/Features/haar.txt').extract(image)
    return hhfe+ haarfe+ MFE #,ehfe

def main():
    a = getExtractors(sys.argv[1])
    print(a)

main()
