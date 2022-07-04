```python

""" webscrape google images and save to new folder """

import os import requests from bs4 import BeautifulSoup from
urllib.request import urlretrieve

search term
===========

search\_term = 'dog'

create folder
=============

folder\_name = search\_term.replace(' ', '*').lower() if not
os.path.exists(folder*name): os.makedirs(folder\_name)

get url
=======

url = "https://www.google.com/search?q=" + search\_term +
"&source=lnms&tbm=isch"

get html
========

header = {'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64)
AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134
Safari/537.36"} html = requests.get(url, headers=header).text

parse html
==========

soup = BeautifulSoup(html, 'html.parser')

get image urls
==============

img\_urls = [] for i in soup.find\_all('div', {'class': 'rg\_meta'}):
link, ext = json.loads(i.text)['ou'], json.loads(i.text)['ity']
img\_urls.append((link, ext))

download images
===============

for i, (img, ext) in enumerate(img\_urls): try: urlretrieve(img,
folder\_name + '/' + search\_term + str(i) + '.' + ext) except:
print('could not download image')




```