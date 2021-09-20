from flask import Flask, render_template, request, send_from_directory
import csv
from collections import namedtuple
from itertools import islice

Article = namedtuple('Article', ['heading', 'image', 'spread_idx'])
DebugImage = namedtuple('DebugImage', ['tag', 'image', 'spread_idx'])

with open('result/index.csv', 'r') as f:
    articles = list(map(Article._make, csv.reader(f)))

with open('result/debug.csv', 'r') as f:
    debug_images = list(map(DebugImage._make, csv.reader(f)))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    q = request.args.get('q')
    result = islice(filter(lambda article: q in article.heading, articles), 0, 25)
    return render_template('search.html', q=q, result=result)

@app.route('/debug')
def debug():
    spread = request.args.get('spread')
    tag = request.args.get('tag')
    result = filter(lambda debug_image: debug_image.spread_idx == spread and debug_image.tag == tag, debug_images)
    return render_template('debug.html', spread=int(spread), tag=tag, result=result)

@app.route('/images/<path:path>')
def send_js(path):
        return send_from_directory('result', path)

if __name__ == "__main__":
    app.run()
