import csv
from csvtuples import Article
import collections

with open('result/index.csv', 'r') as f:
    articles = list(map(Article._make, csv.reader(f)))

print('The number of entries: %d' % len(articles))
print('The number of entries with an empty heading: %d' %
        len([article for article in articles if article.heading == '']))

bad_spreads = [article.spread_idx for article in articles if article.heading == '']
articles = [article for article in articles if article.heading != '']
print('The number of duplicate entries (except empty ones): %d' %
        (len(articles) - len(set(map(lambda article: article.heading, articles)))))

count = 0
for i in range(len(articles) - 1):
    a = articles[i].heading.lower()
    b = articles[i + 1].heading.lower()

    if not (a < b):
        count += 1
        bad_spreads.append(articles[i].spread_idx)

print('The number of entries that are not in dictionary order: %d' % count)

print('\nTop 10 pages with the most entries:')
counter = collections.Counter(bad_spreads)
for spread_idx, count in counter.most_common(10):
    print("  %s: %d entries" % (spread_idx, count))

print('\nBeginning of each alphabet:')
found = set()
print("  a starts at 11")
for spread_idx in range(13, 1224):
    before = collections.Counter([article.heading[:1].lower() for article in articles if article.spread_idx == str(spread_idx - 1)])
    current = collections.Counter([article.heading[:1].lower() for article in articles if article.spread_idx == str(spread_idx)])
    after = collections.Counter([article.heading[:1].lower() for article in articles if article.spread_idx == str(spread_idx + 1)])
    before_char = before.most_common()[0][0]
    after_char = after.most_common()[0][0]
    if ord(before_char) + 1 == ord(after_char) and current[after_char] > 0 and after_char not in found:
        print("  %s starts at %d" % (after_char, spread_idx))
        found.add(after_char)
print("  z starts at 1224")

# TODO: The number of entries with a heading that the spell checker says is wrong
# TODO: The number of cropped columns that have an unusual number of lines
