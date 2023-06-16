import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
import os
import argparse


class HTML(object):
    def __init__(self, title, ncol, save_dir):
        self.title = title
        self.ncol = ncol
        self.save_dir = save_dir
        self.doc = dominate.document(title=title)

    def add_header(self, text):
        """Insert a header to the HTML file

        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, width=400):
        """add images to the HTML file

        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=link):
                                img(style="width:%dpx" % width, src=im)
                            br()
                            p(txt)

    def save(self):
        """save the current content to the HMTL file"""
        html_file = './result_display.html'
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


def create_webpage(result_folder='samples_afhq',
                   display_eval=False,
                   win_size=256,
                   skip_words=None,
                   require_words=None):
    _folders = os.listdir(result_folder)
    if skip_words is not None:
        _folders = [name for name in _folders if not any(skip_word in name for skip_word in skip_words)]
    if require_words is not None:
        _folders = [name for name in _folders if not any(require_word not in name for require_word in require_words)]
    folders = sorted(_folders)
    print(folders)
    ncol = len(folders)
    webpage = HTML(title='Results', ncol=ncol, save_dir=result_folder)

    img_list = list()
    for folder in folders:
        if display_eval:
            img_list += [name for name in os.listdir(os.path.join(result_folder, folder)) if 'jpg' in name]
        else:
            img_list += [name for name in os.listdir(os.path.join(result_folder, folder)) if 'jpg' in name and 'eval' not in name]
    
    img_list = sorted(list(set(img_list)), key=lambda x: int(x.replace('_eval', '')[:-4])+int('eval' in x))
    imgs, txts, links = [], [], []

    for idx in img_list:
        webpage.add_header(f'iter_{idx[:-4]}')
        imgs, txts, links = [], [], []
        for folder in folders:
            imgs.append(os.path.join(result_folder, folder, idx))
            txts.append(folder)
            links.append(os.path.join(result_folder, folder, idx))
        webpage.add_images(imgs, txts, links, width=win_size)
    webpage.save()


parser = argparse.ArgumentParser()

parser.add_argument('--result_folder', type=str, default='samples_afhq')
parser.add_argument('--require_words', type=str, default=None)
parser.add_argument('--skip_words', type=str, default=None)
parser.add_argument('--display_eval', action='store_true')

opt = parser.parse_args()
result_folder = opt.result_folder
display_eval = opt.display_eval
require_words = None if opt.require_words is None else opt.require_words.split(',')
skip_words = None if opt.skip_words is None else opt.skip_words.split(',')
create_webpage(result_folder, display_eval=display_eval,
               require_words=require_words, skip_words=skip_words)
