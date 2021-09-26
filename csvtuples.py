from collections import namedtuple

Article = namedtuple('Article', ['heading', 'image', 'spread_idx'])
ArticleBody = namedtuple('ArticleBody', ['heading', 'spread_idx', 'body'])
DebugImage = namedtuple('DebugImage', ['tag', 'image', 'spread_idx'])
