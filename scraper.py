import os
import sys
import re
import requests
import urllib
from bs4 import BeautifulSoup

def http_request(uri):
    site = 'https://apod.nasa.gov'+uri
    response = requests.get(site)
    return response

def parse_link(link):
# return the link between the 2 ""
# in this case <a href="ap190822.html"> or <a href="image/1908/NGC1499_mosaic.jpg">
    link = link[link.find("\"")+1:]
    return link[:link.find("\"")]

def download(uri, outdir):
    site = 'https://apod.nasa.gov'+uri
    response = requests.get(site)
    index = BeautifulSoup(response.text, 'html.parser')
    lines = str(index).split('\n')
    for line in lines:
        # check for line starting with <a and containing 'image/'
        # in this case: (<a href="image/1908/NGC1499_mosaic.jpg">)
        if line.startswith('<a') and 'image/' in line:
            img_line = line
            break
    else:
        img_line = None

    prev_line = None
    for line in lines:
        # check for line starting with <a and containing html code &lt which stands for '<'
        # in this case: <a href="ap190822.html">&lt;</a>
        if line.startswith('<a') and '&lt' in line:
            prev_line = line
            break

    if prev_line is not None:
        prev_link = parse_link(prev_line)
    else:
        prev_link = None


    if img_line is None:
        print ('Could not parse image link!')
    else:
        img_link = 'https://apod.nasa.gov/apod/'+parse_link(img_line)

        img_name = img_link.rsplit('/', 1)[1]
        outfile = '%s/%s' % (outdir, img_name)
        if os.path.exists(outfile):
            print ('Skipping %s - already exists.' % img_name)
        else:
            print ('Downloading %s' % img_name)
            print (img_link)
            img_data = requests.get(img_link)

            with open(str(outdir+img_name), 'wb') as handler:
                handler.write(img_data.content)

    return prev_link


if __name__=='__main__':
    home_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = home_dir+'/images/'
    prev_link = download('/apod/astropix.html', outdir)
    while prev_link:
        prev_link = download('/apod/%s' % prev_link, outdir)