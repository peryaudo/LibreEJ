import cv2
import csv
from tqdm import tqdm
import multiprocessing
import os
from csvtuples import Article
from csvtuples import ArticleBody
from pytesseract import pytesseract

os.environ['OMP_THREAD_LIMIT'] = '1'
result_dir = 'result'

with open(result_dir + '/index.csv', 'r') as f:
    articles = list(map(Article._make, csv.reader(f)))

def recognize_article_body(article):
    image = cv2.imread(result_dir + '/' + article.image)
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0,0,0))
    image = cv2.bitwise_not(image)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    body = pytesseract.image_to_string(image, lang="eng+jpn")
    return ArticleBody(heading=article.heading, spread_idx=article.spread_idx, body=body)

with open(result_dir + '/body.csv', 'w') as f:
    writer = csv.writer(f)

    pool = multiprocessing.Pool()

    for body in tqdm(pool.imap(recognize_article_body, articles), total=len(articles)):
        writer.writerow(body)
