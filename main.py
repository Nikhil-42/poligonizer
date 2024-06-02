#type: ignore
import cv2
from tqdm import tqdm
import numpy as np
from cv2.typing import MatLike

def pipeline(img: MatLike, k: int, line_width: int=2, threshold_length: float=0.0, debug: bool=False) -> MatLike:
    """Image processing pipeline."""
    
    lines = np.ones_like(img) * 255
    
    # Clean up colors
    # Aggressive guassian blur
    clean = cv2.GaussianBlur(img.copy(), (11, 11), 10)
    
    Z = clean.reshape((-1,3))
    Z = np.float32(Z)
    
    # Uses kmeans to reduce the number of colors present (helps make edges longer)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    compactness,label,centers = cv2.kmeans(Z, k, None, criteria, 10, flags)
    
    centers = np.uint8(centers)
    clean = centers[label.flatten()]
    clean = clean.reshape((img.shape))    

    # Convert to grayscale
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 100)
    
    # Debug edges
    if debug:
        cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
        cv2.imshow('edges', edges)
        cv2.waitKey(0)

    # Find contours
    contours,hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Debug contours
    if debug:
        res = np.zeros_like(img)
        cv2.namedWindow('contours', cv2.WINDOW_NORMAL)
        cv2.imshow('contours', cv2.drawContours(res, contours, -1, (127, 0, 0), 3))
        cv2.waitKey(0)
        
    if debug:
        res = np.zeros_like(img)
    
    # Fit approximate polygons to contours
    for contour in tqdm(contours):
        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True), True)
        
        # Debug approximate polygons
        if debug: 
            cv2.drawContours(res, [approx], -1, (0, 255, 255), 3)

        # Extract lines from polygons and extend them
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
            
            # Draw the lines
            cv2.line(lines, tuple(start.astype(int)), tuple(end.astype(int)), (0, 0, 0), line_width if line_width > 1 else 2)
    
    if debug:
        cv2.namedWindow('approx', cv2.WINDOW_NORMAL)
        cv2.namedWindow('lines', cv2.WINDOW_NORMAL)
        cv2.imshow('approx', res)
        cv2.imshow('lines', lines)
        cv2.waitKey(0)
    
    # Find all the polygons made by the lines
    polygons, _ = cv2.findContours(cv2.cvtColor(lines, cv2.COLOR_BGR2GRAY), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)    
    
    # Debug all polygons
    if debug:
        cv2.namedWindow('polygons', cv2.WINDOW_NORMAL)
        cv2.imshow("polygons", cv2.fillPoly(np.ones_like(lines)*255, polygons, (0, 255, 0)))
        cv2.waitKey(0)
    
    out = np.zeros_like(img)

    # Fill in polygons with color
    mask = np.zeros_like(gray)
    for polygon in tqdm(polygons):
        cv2.fillPoly(mask, [polygon], 255)

        color = cv2.mean(img, mask) 
        out = cv2.fillPoly(out, [polygon], color)
        
        mask[:,:]=0
    
    # Fill in lines if line_width is < 1
    if not line_width > 1:
        out = cv2.dilate(out, np.ones((5-line_width*2, 5-line_width*2), np.uint8), iterations=-(line_width-1) + 1)
    
    # Debug final output
    if debug:
        cv2.namedWindow('out', cv2.WINDOW_NORMAL)
        cv2.imshow('out', out)
        cv2.waitKey(0)
    
    return out 


if __name__ == '__main__':
    import argparse
    import pathlib
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input image file path.')
    parser.add_argument('-o', '--output', type=str, required=False, help='Output image file path (default path/to/input_file + _abstract + .input_ext)')
    parser.add_argument('-k', type=int, default=2, help='Number of colors to reduce to.')
    parser.add_argument('-t', '--threshold_length', type=int, default=0, help='Minimum length of a line to be drawn.')
    parser.add_argument('-l', '--line_width', type=int, default=2, help='Width of the lines to be drawn.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show debug windows.')
    
    args = parser.parse_args()
    
    input_path = pathlib.Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'File {args.input} does not exist.')
    
    if args.output is None:
        args.output = input_path.stem + '_abstract' + input_path.suffix
    
    img = cv2.imread(args.input)
    output = pipeline(img, k=args.k, line_width=args.line_width, threshold_length=args.threshold_length, debug=args.verbose)
    cv2.imwrite(args.output, output)