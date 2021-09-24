import cv2
import imutils
import numpy as np
import re
from pytesseract import pytesseract
import multiprocessing
import csv
import sys
from tqdm import tqdm
import shutil
import os
import argparse
from csvtuples import Article, DebugImage

np.set_printoptions(threshold=np.inf)
os.environ['OMP_THREAD_LIMIT'] = '1'

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
    contours, hierarchy = cv2.findContours(image=hsv_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    h, w = hsv_mask.shape[:2]

    boxes = []
    area_threshold = 0.1
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if cw * ch > (w * h * area_threshold):
            boxes.append([cx, cy, cx + cw, cy + ch])
    boxes = np.asarray(boxes)
    left, top = np.min(boxes, axis=0)[:2]
    right, bottom = np.max(boxes, axis=0)[2:]

    return page[top:bottom,left:right]

def split_column(page):
    h, w = page.shape[:2]
    page = page[int(h * 0.035):,:]
    margin = int(w / 2 * 0.015)
    left_column = page[:,:w//2-margin]
    right_column = page[:,w//2+margin:]
    return left_column, right_column

def leftmost_contour(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return x

def not_too_large_or_too_small(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return w < 50 and h < 50 and w * h > 20 and w > 5

def cut_into_articles(column):
    ys, contours = detect_articles(column)
    if len(ys) == 0:
        return (column, [])
    elif len(ys) == 1:
        return (None, [column])
    articles = []
    h, w = column.shape[:2]
    prev_y = ys[0]
    for y in ys[1:] + [h]:
        articles.append(column[prev_y:y,:])
        prev_y = y
    prev = None
    if ys[0] > 0:
        prev = column[:ys[0],:]
    return (prev, articles)

def detect_lines(column):
    ret, column = cv2.threshold(column, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = column.shape[:2]
    hist = np.sum(column, axis=1) // w > 2
    lower = [y for y in range(h - 1) if hist[y] and not hist[y + 1]]
    upper = [y for y in range(h - 1) if not hist[y] and hist[y + 1]]

    if len(lower) < 2:
        return [], []
    m = np.diff(np.array(lower), n=1).mean()
    lower = [lower[i] for i in range(len(lower) - 1) if lower[i + 1] - lower[i] > m // 2]
    upper = [upper[i] for i in range(len(upper) - 1) if upper[i + 1] - upper[i] > m // 2]
    return lower, upper

def remove_too_close_numbers(src):
    src = sorted(src)
    dst = []
    for num in src:
        if len(dst) == 0:
            dst.append(num)
        elif abs(num - dst[-1]) > 5:
            dst.append(num)
    return dst

def detect_articles(original_column):
    lower, upper = detect_lines(original_column)
    ret, column = cv2.threshold(original_column, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(image=column, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    contours = filter(not_too_large_or_too_small, contours)
    contours = sorted(contours, key=leftmost_contour)
    heading_x = cv2.boundingRect(contours[0])[0]
    # TODO: Stop using heuristic heading tabbing threshold of 10
    contours = list(filter(lambda contour: abs(cv2.boundingRect(contour)[0] - heading_x) < 10, contours))
    heading_ys = []
    for contour in contours:
        contour_y = cv2.boundingRect(contour)[1]
        upper_y = max(filter(lambda y: y <= contour_y, upper), default=contour_y)
        lower_y = max(filter(lambda y: y <= upper_y, lower), default=upper_y)
        heading_ys.append((lower_y + upper_y) // 2)

    heading_ys = remove_too_close_numbers(heading_ys)
    if (len(heading_ys)/len(lower)) > 0.7:
        # It might have failed to detect article headings.
        return [], []
    return heading_ys, contours

def dewarp_column(column):
    h, w = column.shape[:2]

    best = 0
    result = column
    for k in range(2):
        for i in range(10):
            for j in range(10):
                if k == 0:
                    src_pts = np.float32([[0, 0], [0, h], [w, i], [w, h - j]])
                elif k == 1:
                    src_pts = np.float32([[0, i], [0, h - j], [w, 0], [w, h]])
                dst_pts = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
                m = cv2.getPerspectiveTransform(src_pts, dst_pts)
                current = cv2.warpPerspective(column, m, (w, h))
                lower, upper = detect_lines(current)
                if len(lower) > best:
                    best = len(lower)
                    result = current
    return result

def draw_lines(column):
    lower, upper = detect_lines(column)
    ys, contours = detect_articles(column)
    column = cv2.cvtColor(column, cv2.COLOR_GRAY2RGB)
    h, w = column.shape[:2]
    for y in lower:
        cv2.line(column, (0, y), (w - 1, y), (128, 128, 128), 1)
    for y in upper:
        cv2.line(column, (0, y), (w - 1, y), (128, 128, 128), 1)
    for y in ys:
        cv2.line(column, (0, y), (w - 1, y), (0, 255, 0), 1)
    column = cv2.drawContours(column, contours, -1, (0,255,0), 1)
    return column

def recognize_heading(article):
    # https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html#Borders
    article = cv2.copyMakeBorder(article, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0,0,0))
    article = cv2.bitwise_not(article)
    article = cv2.resize(article, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(article, lang="eng", output_type=pytesseract.Output.DICT)
    for text in data['text']:
        if len(text) > 0:
            original_heading = text.split('[')[0]
            return ''.join(c for c in original_heading if c.isalpha())
    return None

def concatenate_articles(articles):
    result = None
    for article in articles:
        if result is None:
            result = article
            continue
        if article is None:
            continue
        rh, rw = result.shape[:2]
        h, w = article.shape[:2]
        if h == 0:
            continue
        if rw > w:
            article = cv2.copyMakeBorder(article, 0, 0, 0, rw - w, cv2.BORDER_CONSTANT, value=(0,0,0))
        elif rw < w:
            result = cv2.copyMakeBorder(result, 0, 0, 0, w - rw, cv2.BORDER_CONSTANT, value=(0,0,0))
        result = np.vstack([result, article])
    return result


def get_articles_from_spread(spread):
    debug_result = []
    debug_result.append(('original', spread.copy()))

    cropped = crop_from_book(spread)
    debug_result.append(('spread', cropped.copy()))

    left_page, right_page = split_left_and_right(cropped)

    spread_prev = None
    result = []
    for page in [left_page, right_page]:
        page = deskew(to_grayscale(crop_page_margin(page)))
        debug_result.append(('page', page.copy()))

        left_column, right_column = split_column(page)
        for column in [left_column, right_column]:
            column = dewarp_column(column)
            debug_result.append(('column', draw_lines(column)))

            prev, articles = cut_into_articles(column)
            if len(result) > 0:
                result[-1] = concatenate_articles([result[-1], prev])
            else:
                spread_prev = concatenate_articles([spread_prev, prev])
            result.extend(articles)
    return (spread_prev, result, debug_result)

def to_grayscale(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.bitwise_not(hsv[...,2])
    return hsv

parser = argparse.ArgumentParser()
parser.add_argument('--debug')
parser.add_argument('--disable_ocr', dest='enable_ocr', action='store_false')
args = parser.parse_args()

# TODO: p. 11 and p. 1225 are shorter irregular pages
page_range = range(12, 1225)
result_dir = 'result'
if args.debug:
    page_range = range(int(args.debug), int(args.debug) + 1)
    result_dir = 'debug'

def save_articles_from_spread(page_idx):
    src = cv2.imread('images/page-%03d.jpg' % page_idx)
    result = []
    debug_result = []
    prev, articles, debug_images = get_articles_from_spread(src)
    prev_filename = None
    if prev is not None:
        prev_filename = 'crop-%d03d-prev.jpg' % page_idx
    cv2.imwrite(result_dir + '/' + prev_filename, prev)
    for i, article in enumerate(articles):
        filename = 'crop-%03d-%d.jpg' % (page_idx, i)
        if args.enable_ocr:
            heading = recognize_heading(article)
        else:
            heading = ''
        cv2.imwrite(result_dir + '/' + filename, article)
        result.append(Article(heading=heading, image=filename, spread_idx=page_idx))
    for i, (tag, image) in enumerate(debug_images):
        filename = 'debug-%03d-%d-%s.jpg' % (page_idx, i, tag)
        cv2.imwrite(result_dir + '/' + filename, image)
        debug_result.append(DebugImage(tag=tag, image=filename, spread_idx=page_idx))
    return (prev_filename, result, debug_result)

if args.debug:
    pool = multiprocessing.Pool(1)
else:
    pool = multiprocessing.Pool()

shutil.rmtree(result_dir + '/', ignore_errors=True)
os.mkdir(result_dir)

with open(result_dir + '/index.csv', 'w') as f:
    with open(result_dir + '/debug.csv', 'w') as debug_f:
        writer = csv.writer(f)
        debug_writer = csv.writer(debug_f)
        last_article = None
        for prev, articles, debug_images in tqdm(pool.imap(save_articles_from_spread, page_range), total=len(page_range)):
            if prev is not None and last_article is not None:
                cv2.imwrite(
                        result_dir + '/' + last_article.image,
                        concatenate_articles([cv2.imread(result_dir + '/' + last_article.image), cv2.imread(result_dir + '/' + prev)]))
                os.remove(result_dir + '/' + prev)

            for article in articles:
                last_article = article
                writer.writerow(article)
            for debug_image in debug_images:
                debug_writer.writerow(debug_image)

if args.debug:
    print('Run')
    print('    python3 web.py --debug')
    print('and check')
    print('    http://localhost:5000/debug?spread=%d&tag=column' % int(args.debug))
