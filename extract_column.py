import cv2
import imutils
import numpy as np
import re
from pytesseract import pytesseract
import multiprocessing

np.set_printoptions(threshold=np.inf)

def fill_from_corners(gray):
    h, w = gray.shape[:2]
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    ret, gray, mask, rect = cv2.floodFill(gray, mask, (0, 0), 0)
    ret, gray, mask, rect = cv2.floodFill(gray, mask, (w - 1, 0), 0)
    ret, gray, mask, rect = cv2.floodFill(gray, mask, (0, h - 1), 0)
    ret, gray, mask, rect = cv2.floodFill(gray, mask, (w - 1, h - 1), 0)
    return gray

def crop_from_book(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array([15, 0, 230])
    hsv_upper = np.array([20, 255, 255])
    hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel, iterations=4)

    x, y, w, h = cv2.boundingRect(hsv_mask)
    img = img[y:y+h,x:x+w]
    hsv = hsv[y:y+h,x:x+w]

    hsv_lower = np.array([0, 0, 0])
    hsv_upper = np.array([179, 80, 150])
    hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    hsv_mask = cv2.dilate(hsv_mask, kernel, iterations = 4)
    contours, hierarchy = cv2.findContours(image=hsv_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # For debug
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    boxes = []
    area_threshold = 0.1
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if cw * ch > (w * h * area_threshold):
            boxes.append([cx, cy, cx + cw, cy + ch])
    boxes = np.asarray(boxes)
    left, top = np.min(boxes, axis=0)[:2]
    right, bottom = np.max(boxes, axis=0)[2:]

    return img[top:bottom,left:right]

def split_left_and_right(cropped):
    h, w = cropped.shape[:2]
    margin = int(w / 2 * 0.01)
    left_page = cropped[:,:w//2 - margin]
    right_page = cropped[:,w//2 + margin:]
    return left_page, right_page

def deskew(original_page):
    ret, page = cv2.threshold(original_page, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    page = fill_from_corners(page)
    angle = -1.5
    best_angle = angle
    best_score = 0
    while angle <= 1.5:
        rotated = imutils.rotate(page, angle)
        hist = np.sum(rotated, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        if score > best_score:
            best_score = score
            best_angle = angle
        angle += 0.01
    return imutils.rotate(original_page, best_angle)

def crop_page_margin(page):
    hsv = cv2.cvtColor(page, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array([0, 0, 0])
    hsv_upper = np.array([179, 80, 150])
    hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    hsv_mask = cv2.dilate(hsv_mask, kernel, iterations = 4)

    x, y, w, h = cv2.boundingRect(hsv_mask)

    return page[y:y+h,x:x+w]

def split_column(page):
    h, w = page.shape[:2]
    page = page[int(h * 0.035):,:]
    left_column = page[:,:w//2]
    right_column = page[:,w//2:]
    return left_column, right_column

def leftmost_contour(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return x

def not_too_long(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return w < 100 and h < 100

def detect_article(original_column):
    ret, column = cv2.threshold(column, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(image=column, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    contours = filter(not_too_long, contours)
    contours = sorted(contours, key=leftmost_contour)
    heading_x = cv2.boundingRect(contours[0])[0]
    # TODO: Stop using heuristic heading tabbing threshold of 5
    contours = filter(lambda contour: abs(cv2.boundingRect(contour)[0] - heading_x) < 5, contours)
    # TODO: Stop subtracting 5 and use a smarter method
    heading_ys = map(lambda contour: cv2.boundingRect(contour)[1] - 5, contours)
    return list(sorted(list(set(heading_ys))))

def cut_into_articles(column, ys):
    articles = []
    h, w = column.shape[:2]
    prev_y = ys[0]
    # TODO: Stop dropping ys[0] and concatenate it with an article in the previous column
    for y in ys[1:] + [h]:
        articles.append(column[prev_y:y,:])
        prev_y = y
    return articles

def recognize_heading(article):
    article = cv2.bitwise_not(article)
    article = cv2.resize(article, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(article, lang="eng", output_type=pytesseract.Output.DICT)
    for text in data['text']:
        if len(text) > 0:
            return text.split('[')[0]
    return None

def get_articles_from_spread(spread):
    cropped = crop_from_book(spread)
    left_page, right_page = split_left_and_right(cropped)
    result = []
    for page in [left_page, right_page]:
        page = deskew(to_grayscale(crop_page_margin(page)))
        left_column, right_column = split_column(page)
        for column in [left_column, right_column]:
            heading_ys = detect_article(column)
            articles = cut_into_articles(column, heading_ys)
            for article in articles:
                result.append(article)
    return result

def to_grayscale(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.bitwise_not(hsv[...,2])
    return hsv

# TODO: p. 11 and p. 1225 are shorter irregular pages
page_range = range(12, 1225)

def test_page_split(page_idx):
    src = cv2.imread('images/page-%03d.jpg' % page_idx)
    try:
        dst = crop_from_book(src)
        dst_left, dst_right = split_left_and_right(dst)
        dst_left = crop_page_margin(dst_left)
        dst_right = crop_page_margin(dst_right)
        dst_left = to_grayscale(dst_left)
        dst_right = to_grayscale(dst_right)
        dst_left = deskew(dst_left)
        dst_right = deskew(dst_right)

        dst_left_1, dst_left_2 = split_column(dst_left)
        dst_right_1, dst_right_2 = split_column(dst_right)
    except Exception as e:
        print('error while processing', page_idx, e)
    cv2.imwrite(('cropped/crop-%03d-left-1.jpg' % page_idx), dst_left_1)
    cv2.imwrite(('cropped/crop-%03d-left-2.jpg' % page_idx), dst_left_2)
    cv2.imwrite(('cropped/crop-%03d-right-1.jpg' % page_idx), dst_right_1)
    cv2.imwrite(('cropped/crop-%03d-right-2.jpg' % page_idx), dst_right_2)

# pool = multiprocessing.Pool()
# pool.map(test_page_split, page_range)
for page_idx in range(12, 50):
    test_page_split(page_idx)

# img = cv2.imread("images/page-015.jpg")
# img = cv2.imread("jpegOutput.jpg")

# for article in get_articles_from_spread(img):
#     print(recognize_heading(article))
#     cv2.imshow('image', article)
#     cv2.waitKey(0)
