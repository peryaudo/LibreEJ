from flask import Flask, render_template, request, send_from_directory
import csv
from itertools import islice
from csvtuples import Article, DebugImage
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if args.debug:
    result_dir = 'debug'
else:
    result_dir = 'result'

with open(result_dir + '/index.csv', 'r') as f:
    articles = list(map(Article._make, csv.reader(f)))

with open(result_dir + '/debug.csv', 'r') as f:
    debug_images = list(map(DebugImage._make, csv.reader(f)))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    q = request.args.get('q').strip().lower()
    result = islice(filter(lambda article: q in article.heading.lower(), articles), 0, 25)
    return render_template('search.html', q=q, result=result)

@app.route('/debug')
def debug():
    spread = request.args.get('spread')
    tag = request.args.get('tag')
    result = []
    debug_result = []
    if tag == 'article':
        result = filter(lambda article: article.spread_idx == spread, articles)
    else:
        debug_result = filter(lambda debug_image: debug_image.spread_idx == spread and debug_image.tag == tag, debug_images)
    return render_template('debug.html', spread=int(spread), tag=tag, debug_result=debug_result, result=result)

@app.route('/images/<path:path>')
def send_js(path):
        return send_from_directory(result_dir, path)

if __name__ == "__main__":
    app.run()
