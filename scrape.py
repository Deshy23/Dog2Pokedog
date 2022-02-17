from bs4 import *
import requests
import os
from io import BytesIO
from PIL import Image

def main():
    dogs = ['arcanine', 'growlithe', 'ninetales', 'vulpix', 'granbull', 'houndoom', 'houndour', 'smeargle', 'snubbull', 'electrike', 'manectric', 'mightyena', 'nickit', 'poochyena', 'thievul', 'herdier', 
        'lillipup', 'stoutland', 'zorua', 'zoroark', 'braixen', 'delphox', 'fennekin', 'furfrou', 'boltund', 'lycanroc', 'rockruff', 'yamper','zacian', 'zamazenta', 'shinx', 'luxio', 'luxray', 'eevee', 'vaporeon',
        'jolteon', 'flareon', 'espeon', 'umbreon', 'leafeon', 'glaceon', 'sylveon' 'zigzagoon', 'absol', 'riolu', 'lucario', 'entei', 'raikou, suicune']
    count = [0 for i in range(len(dogs))]
    next, count =  get_images('https://archives.bulbagarden.net/w/index.php?title=Category:Anime_screenshots&fileuntil=AG032+Treat.png%0AAG032+Treat.png#mw-category-media', count, dogs)
    i = 0
    while(True):
        next, count = get_images('https://archives.bulbagarden.net' + next, count, dogs)
        i += 1
        print(str(i) + 'th page')

def extract_image(img, name):
    mid =  requests.get('https://archives.bulbagarden.net' + img['href'])
    soup = BeautifulSoup(mid.text, 'html.parser')
    im = soup.find_all('a', class_ = 'internal')
    image =  requests.get(im[0]['href']).content
    im = Image.open(BytesIO(image))
    im.save(name + '.png', quality = 95)

def get_images(url, count, dogs):
    r =  requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    imgs = soup.find_all(class_ = 'galleryfilename galleryfilename-truncate')
    for img in imgs:
        for i in range(len(count)):
            if dogs[i] in str(img).lower():
                extract_image(img, dogs[i] + str(count[i]))
                count[i] += 1
                print(count)
    x  = soup.find_all('a')
    for link in x:
        if 'next page' in link:
            return(link['href'], count)

if __name__ == '__main__':
    main()
