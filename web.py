from flask import Flask, render_template, request, send_from_directory
import csv
from itertools import islice
from csvtuples import Article, ArticleBody, DebugImage

result_dir = 'result'

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.debug:
        result_dir = 'debug'

with open(result_dir + '/index.csv', 'r') as f:
    articles = list(map(Article._make, csv.reader(f)))

try:
    with open(result_dir + '/body.csv', 'r') as f:
        bodies = list(map(ArticleBody._make, csv.reader(f)))
except:
    bodies = [ArticleBody(heading="", spread_idx=-1, body="")] * len(articles)

with open(result_dir + '/debug.csv', 'r') as f:
    debug_images = list(map(DebugImage._make, csv.reader(f)))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    q = request.args.get('q').strip().lower()
    result = filter(lambda article: q in article[0].heading.lower(), zip(articles, bodies))
    result = sorted(result, key=lambda article: article[0].heading.lower().find(q))
    result = islice(result, 0, 25)
    return render_template('search.html', q=q, result=result)

@app.route('/debug')
def debug():
    spread = request.args.get('spread')
    tag = request.args.get('tag')
    result = []
    debug_result = []
    if tag == 'article':
        result = filter(lambda article: article[0].spread_idx == spread, zip(articles, bodies))
    else:
        debug_result = filter(lambda debug_image: debug_image.spread_idx == spread and debug_image.tag == tag, debug_images)
    return render_template('debug.html', spread=int(spread), tag=tag, debug_result=debug_result, result=result)

@app.route('/bad')
def bad():
    mode = request.args.get('mode')
    if mode == 'empty':
        result = list(filter(lambda article: article.heading == '', articles))
    else:
        result = []
        for i in range(len(articles) - 1):
            a = articles[i].heading.lower()
            b = articles[i + 1].heading.lower()
            if not (a < b):
                result.append(articles[i])
    return render_template('bad.html', total=len(result), result=result[:200])

@app.route('/images/<path:path>')
def send_js(path):
        return send_from_directory(result_dir, path)

if __name__ == "__main__":
    app.run()
