# type: ignore

from math import e
import cv2
import numpy as np
from cv2.typing import MatLike

def pipeline(img: MatLike, k: int, threshold_length: float) -> MatLike:
    """Image processing pipeline."""
    
    # Aggressive guassian blur
    img = cv2.GaussianBlur(img, (11, 11), 10)
    
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    compactness,label,centers = cv2.kmeans(Z, k, None, criteria, 10, flags)
    
    centers = np.uint8(centers)
    res = centers[label.flatten()]
    res = res.reshape((img.shape))    
  
    thresholds = np.empty((k, *img.shape[:2]), dtype=np.uint8)
    for i, center in enumerate(centers):
        cv2.inRange(res, center, center, thresholds[i])
        cv2.imwrite(f'output_{i}.jpg', thresholds[i])
    
    for thresh in thresholds:
        contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(res, contours, -1, (127, 0, 0), 3)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.001*cv2.arcLength(contour,True),True)

            for i1 in range(len(approx)):
                i2 = (i1+1)%len(approx)
                p1 = approx[i1][0]
                p2 = approx[i2][0]
                r = p2 - p1
                
                if np.linalg.norm(r) < threshold_length:
                    continue
            
                height = img.shape[0]
                width = img.shape[1]
                
                if r[0] < 0:
                    p1, p2, r = p2, p1, -r
                
                if r[0] == 0:
                    start = np.array([p1[0], 0])
                    end = np.array([p1[0], height])
                elif r[1] == 0:
                    start = np.array([0, p1[1]])
                    end = np.array([width, p1[1]])
                else:
                    y_target = height if r[1] < 0 else 0
                    
                    t = min((p1[0] - 0)/r[0], (p1[1] - y_target)/r[1])
                    start = p1 - t * r
                    
                    y_target = 0 if r[1] < 0 else img.shape[0]
                    t = max((p1[0] - img.shape[1])/r[0], (p1[1] - y_target)/r[1])
                    end = p1 - t * r
                    
                cv2.line(res, tuple(start.astype(int)), tuple(end.astype(int)), (0, 0, 0), 1)
                cv2.drawContours(res, [approx], -1, (0, 255, 255), 3)
    return res


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('-o', '--output', type=str, default='output.jpg')
    parser.add_argument('-k', type=int, default=2)
    parser.add_argument('-t', '--threshold_length', type=int, default=100)
    
    args = parser.parse_args()
    
    img = cv2.imread(args.input)
    output = pipeline(img, k=args.k, threshold_length=args.threshold_length)
    cv2.imwrite(args.output, output)