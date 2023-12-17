# type: ignore

from math import e
from tqdm import tqdm
import cv2
import numpy as np
from cv2.typing import MatLike

def pipeline(img: MatLike, k: int, line_width = 2, threshold_length = 0.0) -> MatLike:
    """Image processing pipeline."""
    
    lines = np.ones_like(img) * 255
    
    # Clean up colors
    # Aggressive guassian blur
    clean = cv2.GaussianBlur(img.copy(), (11, 11), 10)
    
    Z = clean.reshape((-1,3))
    Z = np.float32(Z)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    compactness,label,centers = cv2.kmeans(Z, k, None, criteria, 10, flags)
    
    centers = np.uint8(centers)
    clean = centers[label.flatten()]
    clean = clean.reshape((img.shape))    

    # Convert to grayscale
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50,100)
    cv2.imwrite('edges.jpg', edges)

    contours,hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # img = cv2.drawContours(res, contours, -1, (127, 0, 0), 3)
    
    for contour in tqdm(contours):
        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True),True)
        # cv2.drawContours(res, [approx], -1, (0, 255, 255), 3)

        for i1 in range(len(approx)):
            i2 = (i1+1)%len(approx)
            p1 = approx[i1][0]
            p2 = approx[i2][0]
            r = p2 - p1
            
            if np.linalg.norm(r) < threshold_length:
                continue
        
            height = gray.shape[0]
            width = gray.shape[1]
            
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
                
                y_target = 0 if r[1] < 0 else gray.shape[0]
                t = max((p1[0] - gray.shape[1])/r[0], (p1[1] - y_target)/r[1])
                end = p1 - t * r
                
            # cv2.line(res, p1, p2, (0, 255, 0), 2)
            cv2.line(lines, tuple(start.astype(int)), tuple(end.astype(int)), (0, 0, 0), 2)
    
    cv2.imwrite('lines.jpg', lines)
    polygons, heiarchy = cv2.findContours(cv2.cvtColor(lines, cv2.COLOR_BGR2GRAY), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)    
    
    # Debug all polygons
    cv2.imwrite("polygons.jpg", cv2.fillPoly(np.array(lines), polygons, (0, 255, 0)))
    
    out = np.zeros_like(img)

    mask = np.zeros_like(gray)
    for polygon in tqdm(polygons):
        cv2.fillPoly(mask, [polygon], 255)

        color = cv2.mean(img, mask) 
        out = cv2.fillPoly(out, [polygon], color)
        
        mask[:,:]=0
             
    return out 


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('-o', '--output', type=str, default='output.jpg')
    parser.add_argument('-k', type=int, default=2)
    parser.add_argument('-t', '--threshold_length', type=int, default=0)
    parser.add_argument('-l', '--line_width', type=int, default=2)
    
    args = parser.parse_args()
    
    img = cv2.imread(args.input)
    output = pipeline(img, k=args.k, line_width=args.line_width, threshold_length=args.threshold_length)
    cv2.imwrite(args.output, output)