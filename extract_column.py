import cv2
import imutils
import numpy as np
import re
from pytesseract import pytesseract

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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = gray.shape[:2]
    gray = fill_from_corners(gray)

    book_threshold = 145

    vertical = np.sum(gray, axis=0)//w > book_threshold
    horizontal = np.sum(gray, axis=1)//h > book_threshold

    vertical_indexes = np.where(vertical == True)
    horizontal_indexes = np.where(horizontal == True)

    v_begin = vertical_indexes[0][0]
    v_end = vertical_indexes[0][-1]

    h_begin = horizontal_indexes[0][0]
    h_end = horizontal_indexes[0][-1]

    return img[h_begin:h_end, v_begin:v_end]

def split_left_and_right(cropped):
    h, w = cropped.shape[:2]
    margin = int((w // 2)* 0.02)
    left_page = cropped[:,:w//2 - margin]
    right_page = cropped[:,w//2 + margin:]
    return left_page, right_page

def deskew(original_page):
    page = cv2.cvtColor(original_page, cv2.COLOR_BGR2GRAY)
    ret, page = cv2.threshold(page, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = gray.shape[:2]
    gray = fill_from_corners(gray)
    # TODO: vertical and horizontal are opposite
    vertical = np.sum(gray, axis=0)//w > 0
    horizontal = np.sum(gray, axis=1)//h > 40

    vertical_indexes = np.where(vertical == True)
    horizontal_indexes = np.where(horizontal == True)

    v_begin = vertical_indexes[0][0]
    v_end = vertical_indexes[0][-1] + int(h * 0.005)

    h_begin = horizontal_indexes[0][0]
    h_end = horizontal_indexes[0][-1] + int(h * 0.005)

    return page[h_begin:h_end, v_begin:v_end]

def split_column(page):
    h, w = page.shape[:2]
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
    column = cv2.cvtColor(original_column, cv2.COLOR_BGR2GRAY)
    ret, column = cv2.threshold(column, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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
    article = cv2.cvtColor(article, cv2.COLOR_BGR2GRAY)
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
        page = crop_page_margin(deskew(page))
        left_column, right_column = split_column(page)
        for column in [left_column, right_column]:
            heading_ys = detect_article(column)
            articles = cut_into_articles(column, heading_ys)
            for article in articles:
                result.append(article)
    return result

# TODO: p. 11 and p. 1225 are shorter irregular pages
page_range = range(12, 1225)

def crop_from_book2(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array([15, 0, 230])
    hsv_upper = np.array([20, 255, 255])
    hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    x, y, w, h = cv2.boundingRect(hsv_mask)

    # hsv = hsv[y:y+h,x:x+w]
    # hsv_lower = np.array([0, 0, 0])
    # hsv_upper = np.array([179, 50, 150])
    # hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # hsv_mask = cv2.dilate(hsv_mask, kernel, iterations = 4)

    # cv2.imshow('image', hsv_mask)
    # cv2.waitKey(0)
    return img[y:y+h,x:x+w]

for page_idx in page_range:
    src_filename = 'images/page-%03d.jpg' % page_idx
    dst_filename = 'cropped/crop-%03d.jpg' % page_idx
    src = cv2.imread(src_filename)
    try:
        dst = crop_from_book2(src)
    except Exception as e:
        print('error while processing', src_filename, e)
    cv2.imwrite(dst_filename, dst)

# img = cv2.imread("images/page-015.jpg")
# img = cv2.imread("jpegOutput.jpg")

# for article in get_articles_from_spread(img):
#     print(recognize_heading(article))
#     cv2.imshow('image', article)
#     cv2.waitKey(0)
