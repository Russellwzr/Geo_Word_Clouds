# Author: Andreas Christian Mueller <t3kcit@gmail.com>
#
# (c) 2012
# Modified by: Paul Nechifor <paul@nechifor.net>
#
# License: MIT

from __future__ import division

import warnings
from random import Random
import os
import re
import sys
import colorsys
import numpy as np
import queue as Q
import cmath
from operator import itemgetter

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageFont

from .query_integral_image import query_integral_image
from .tokenization import unigrams_and_bigrams, process_tokens


FILE = os.path.dirname(__file__)
FONT_PATH = os.environ.get('FONT_PATH', os.path.join(FILE, 'DroidSansMono.ttf'))
STOPWORDS = set(map(str.strip, open(os.path.join(FILE, 'stopwords')).readlines()))

class WordCluster(object):
    def __init__(self,word,x,y,num,direction,min_x,max_x,min_y,max_y,idx):
        self.word = word
        self.x = x
        self.y = y
        self.num = num
        self.direction = direction
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.idx = idx

    def __lt__(self, other):
        return self

    def __lt__(self, other):
        return self.num > other.num

    def __str__(self):
        return '(' + self.word + ', ' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.num) + ', '+str(self.direction)+')'


class IntegralOccupancyMap(object):
    def __init__(self, height, width, mask):
        self.height = height
        self.width = width
        if mask is not None:
            # the order of the cumsum's is important for speed ?!
            self.integral = np.cumsum(np.cumsum(255 * mask, axis=1),
                                      axis=0).astype(np.uint32)
        else:
            self.integral = np.zeros((height, width), dtype=np.uint32)

    def sample_position(self, size_x, size_y, random_state):
        return query_integral_image(self.integral, size_x, size_y,
                                    random_state)

    def sample_position_v2(self, size_x, size_y, b_x, b_y, e_x, e_y, center_x, center_y):
        dis_to_center = 99999999
        ans_x = 0
        ans_y = 0
        for y in range(max(1, b_y), min(e_y - size_y, self.height - size_y)):
            for x in range(max(1,b_x), min(e_x - size_x,self.width - size_x)):
                area = self.integral[y - 1, x - 1] + self.integral[y + size_y - 1, x + size_x - 1]
                area -= self.integral[y - 1, x + size_x - 1] + self.integral[y + size_y - 1, x - 1]
                if not area:
                    tmpdis = Euclidean_distance(x + size_x/2 , center_x, y + size_y/2, center_y).real
                    if tmpdis < dis_to_center:
                        dis_to_center = tmpdis
                        ans_x = x
                        ans_y = y
        if dis_to_center == 99999999:
            return None
        else:
            return ans_y, ans_x


    def update(self, img_array, pos_x, pos_y):
        partial_integral = np.cumsum(np.cumsum(img_array[pos_x:, pos_y:],
                                               axis=1), axis=0)
        # paste recomputed part into old image
        # if x or y is zero it is a bit annoying
        if pos_x > 0:
            if pos_y > 0:
                partial_integral += (self.integral[pos_x - 1, pos_y:]
                                     - self.integral[pos_x - 1, pos_y - 1])
            else:
                partial_integral += self.integral[pos_x - 1, pos_y:]
        if pos_y > 0:
            partial_integral += self.integral[pos_x:, pos_y - 1][:, np.newaxis]

        self.integral[pos_x:, pos_y:] = partial_integral


def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    """Random hue color generation.

    Default coloring method. This just picks a random hue with value 80% and
    lumination 50%.

    Parameters
    ----------
    word, font_size, position, orientation  : ignored.

    random_state : random.Random object or None, (default=None)
        If a random object is given, this is used for generating random
        numbers.

    """
    if random_state is None:
        random_state = Random()
    return "hsl(%d, 80%%, 50%%)" % random_state.randint(0, 255)


class colormap_color_func(object):
    """Color func created from matplotlib colormap.

    Parameters
    ----------
    colormap : string or matplotlib colormap
        Colormap to sample from

    Example
    -------
    >>> WordCloud(color_func=colormap_color_func("magma"))

    """
    def __init__(self, colormap):
        import matplotlib.pyplot as plt
        self.colormap = plt.cm.get_cmap(colormap)

    def __call__(self, word, font_size, position, orientation,
                 random_state=None, **kwargs):
        if random_state is None:
            random_state = Random()
        r, g, b, _ = np.maximum(0, 255 * np.array(self.colormap(
            random_state.uniform(0, 1))))
        return "rgb({:.0f}, {:.0f}, {:.0f})".format(r, g, b)


def get_single_color_func(color):
    """Create a color function which returns a single hue and saturation with.
    different values (HSV). Accepted values are color strings as usable by
    PIL/Pillow.

    >>> color_func1 = get_single_color_func('deepskyblue')
    >>> color_func2 = get_single_color_func('#00b4d2')
    """
    old_r, old_g, old_b = ImageColor.getrgb(color)
    rgb_max = 255.
    h, s, v = colorsys.rgb_to_hsv(old_r / rgb_max, old_g / rgb_max,
                                  old_b / rgb_max)

    def single_color_func(word=None, font_size=None, position=None,
                          orientation=None, font_path=None, random_state=None):
        """Random color generation.

        Additional coloring method. It picks a random value with hue and
        saturation based on the color given to the generating function.

        Parameters
        ----------
        word, font_size, position, orientation  : ignored.

        random_state : random.Random object or None, (default=None)
          If a random object is given, this is used for generating random
          numbers.

        """
        if random_state is None:
            random_state = Random()
        r, g, b = colorsys.hsv_to_rgb(h, s, random_state.uniform(0.2, 1))
        return 'rgb({:.0f}, {:.0f}, {:.0f})'.format(r * rgb_max, g * rgb_max,
                                                    b * rgb_max)
    return single_color_func


def Euclidean_distance(x1, x2, y1, y2):
    return cmath.sqrt((x2 - x1) * (x2 - x1)  + (y2 - y1) * (y2 - y1))

def Hausdorff_Distance(x, y, w, h, idx, cluster_points):
    sz = len(cluster_points[idx])
    min_x = x
    max_x = x + w
    min_y = y
    max_y = y + h
    dis = 0
    num = 0
    for i in range(0, sz):
        cur_x = cluster_points[idx][i][0]
        cur_y = cluster_points[idx][i][1]
        if cur_x >= min_x and cur_x <= max_x and cur_y >= min_y and cur_y <= max_y:
            num = num + 1
            continue
        mindis = 99999999
        for x2 in range(min_x,max_x + 1):
            tmpdis1 = Euclidean_distance(cur_x, x2, cur_y, min_y).real
            tmpdis2 = Euclidean_distance(cur_x, x2, cur_y, max_y).real
            tmpdis = min(tmpdis1, tmpdis2)
            if mindis > tmpdis:
                mindis = tmpdis
        for y2 in range(min_y, max_y + 1):
            tmpdis1 = Euclidean_distance(cur_x, min_x, cur_y, y2).real
            tmpdis2 = Euclidean_distance(cur_x, max_x, cur_y, y2).real
            tmpdis = min(tmpdis1, tmpdis2)
            if mindis > tmpdis:
                mindis = tmpdis
        dis = dis + mindis
    return dis,num

def Hausdorff_DistanceVer2(x, y, w, h, cur_x,cur_y):
    min_x = x
    max_x = x + w
    min_y = y
    max_y = y + h
    dis = 0
    if cur_x >= min_x and cur_x <= max_x and cur_y >= min_y and cur_y <= max_y:
        return 0
    else:
        mindis = 99999999
        for x2 in range(min_x,max_x + 1):
            tmpdis1 = Euclidean_distance(cur_x, x2, cur_y, min_y).real
            tmpdis2 = Euclidean_distance(cur_x, x2, cur_y, max_y).real
            tmpdis = min(tmpdis1, tmpdis2)
            if mindis > tmpdis:
                mindis = tmpdis
        for y2 in range(min_y, max_y + 1):
            tmpdis1 = Euclidean_distance(cur_x, min_x, cur_y, y2).real
            tmpdis2 = Euclidean_distance(cur_x, max_x, cur_y, y2).real
            tmpdis = min(tmpdis1, tmpdis2)
            if mindis > tmpdis:
                mindis = tmpdis
        dis = mindis
    return dis


def Fill_Map(MM, x, y, size_x, size_y):
    for i in range(x - 1, x+size_x):
        for j in range(y - 1, y+size_y):
            MM[i][j] = 1

class WordCloud(object):
    r"""Word cloud object for generating and drawing.

    Parameters
    ----------
    font_path : string
        Font path to the font that will be used (OTF or TTF).
        Defaults to DroidSansMono path on a Linux machine. If you are on
        another OS or don't have this font, you need to adjust this path.

    width : int (default=400)
        Width of the canvas.

    height : int (default=200)
        Height of the canvas.

    prefer_horizontal : float (default=0.90)
        The ratio of times to try horizontal fitting as opposed to vertical.
        If prefer_horizontal < 1, the algorithm will try rotating the word
        if it doesn't fit. (There is currently no built-in way to get only
        vertical words.)

    mask : nd-array or None (default=None)
        If not None, gives a binary mask on where to draw words. If mask is not
        None, width and height will be ignored and the shape of mask will be
        used instead. All white (#FF or #FFFFFF) entries will be considerd
        "masked out" while other entries will be free to draw on. [This
        changed in the most recent version!]

    contour_width: float (default=0)
        If mask is not None and contour_width > 0, draw the mask contour.

    contour_color: color value (default="black")
        Mask contour color.

    scale : float (default=1)
        Scaling between computation and drawing. For large word-cloud images,
        using scale instead of larger canvas size is significantly faster, but
        might lead to a coarser fit for the words.

    min_font_size : int (default=4)
        Smallest font size to use. Will stop when there is no more room in this
        size.

    font_step : int (default=1)
        Step size for the font. font_step > 1 might speed up computation but
        give a worse fit.

    max_words : number (default=200)
        The maximum number of words.

    stopwords : set of strings or None
        The words that will be eliminated. If None, the build-in STOPWORDS
        list will be used. Ignored if using generate_from_frequencies.

    background_color : color value (default="black")
        Background color for the word cloud image.

    max_font_size : int or None (default=None)
        Maximum font size for the largest word. If None, height of the image is
        used.

    mode : string (default="RGB")
        Transparent background will be generated when mode is "RGBA" and
        background_color is None.

    relative_scaling : float (default='auto')
        Importance of relative word frequencies for font-size.  With
        relative_scaling=0, only word-ranks are considered.  With
        relative_scaling=1, a word that is twice as frequent will have twice
        the size.  If you want to consider the word frequencies and not only
        their rank, relative_scaling around .5 often looks good.
        If 'auto' it will be set to 0.5 unless repeat is true, in which
        case it will be set to 0.

        .. versionchanged: 2.0
            Default is now 'auto'.

    color_func : callable, default=None
        Callable with parameters word, font_size, position, orientation,
        font_path, random_state that returns a PIL color for each word.
        Overwrites "colormap".
        See colormap for specifying a matplotlib colormap instead.
        To create a word cloud with a single color, use
        ``color_func=lambda *args, **kwargs: "white"``.
        The single color can also be specified using RGB code. For example
        ``color_func=lambda *args, **kwargs: (255,0,0)`` sets color to red.

    regexp : string or None (optional)
        Regular expression to split the input text into tokens in process_text.
        If None is specified, ``r"\w[\w']+"`` is used. Ignored if using
        generate_from_frequencies.

    collocations : bool, default=True
        Whether to include collocations (bigrams) of two words. Ignored if using
        generate_from_frequencies.


        .. versionadded: 2.0

    colormap : string or matplotlib colormap, default="viridis"
        Matplotlib colormap to randomly draw colors from for each word.
        Ignored if "color_func" is specified.

        .. versionadded: 2.0

    normalize_plurals : bool, default=True
        Whether to remove trailing 's' from words. If True and a word
        appears with and without a trailing 's', the one with trailing 's'
        is removed and its counts are added to the version without
        trailing 's' -- unless the word ends with 'ss'. Ignored if using
        generate_from_frequencies.

    repeat : bool, default=False
        Whether to repeat words and phrases until max_words or min_font_size
        is reached.

    Attributes
    ----------
    ``words_`` : dict of string to float
        Word tokens with associated frequency.

        .. versionchanged: 2.0
            ``words_`` is now a dictionary

    ``layout_`` : list of tuples (string, int, (int, int), int, color))
        Encodes the fitted word cloud. Encodes for each word the string, font
        size, position, orientation and color.

    Notes
    -----
    Larger canvases with make the code significantly slower. If you need a
    large word cloud, try a lower canvas size, and set the scale parameter.

    The algorithm might give more weight to the ranking of the words
    than their actual frequencies, depending on the ``max_font_size`` and the
    scaling heuristic.
    """

    def __init__(self, font_path=None, width=400, height=200, margin=2,
                 ranks_only=None, prefer_horizontal=.9, mask=None, scale=1,
                 color_func=None, max_words=200, min_font_size=4,
                 stopwords=None, random_state=None, background_color='black',
                 max_font_size=None, font_step=1, mode="RGB",
                 relative_scaling='auto', regexp=None, collocations=True,
                 colormap=None, normalize_plurals=True, contour_width=0,
                 contour_color='black', repeat=False):
        if font_path is None:
            font_path = FONT_PATH
        if color_func is None and colormap is None:
            # we need a color map
            import matplotlib
            version = matplotlib.__version__
            if version[0] < "2" and version[2] < "5":
                colormap = "hsv"
            else:
                colormap = "viridis"
        self.colormap = colormap
        self.collocations = collocations
        self.font_path = font_path
        self.width = width
        self.height = height
        self.margin = margin
        self.prefer_horizontal = prefer_horizontal
        self.mask = mask
        self.contour_color = contour_color
        self.contour_width = contour_width
        self.scale = scale
        self.color_func = color_func or colormap_color_func(colormap)
        self.max_words = max_words
        self.stopwords = stopwords if stopwords is not None else STOPWORDS
        self.min_font_size = min_font_size
        self.font_step = font_step
        self.regexp = regexp
        if isinstance(random_state, int):
            random_state = Random(random_state)
        self.random_state = random_state
        self.background_color = background_color
        self.max_font_size = max_font_size
        self.mode = mode

        if relative_scaling == "auto":
            if repeat:
                relative_scaling = 0
            else:
                relative_scaling = .5

        if relative_scaling < 0 or relative_scaling > 1:
            raise ValueError("relative_scaling needs to be "
                             "between 0 and 1, got %f." % relative_scaling)
        self.relative_scaling = relative_scaling
        if ranks_only is not None:
            warnings.warn("ranks_only is deprecated and will be removed as"
                          " it had no effect. Look into relative_scaling.",
                          DeprecationWarning)
        self.normalize_plurals = normalize_plurals
        self.repeat = repeat

    def fit_words(self, frequencies):
        """Create a word_cloud from words and frequencies.

        Alias to generate_from_frequencies.

        Parameters
        ----------
        frequencies : dict from string to float
            A contains words and associated frequency.

        Returns
        -------
        self
        """
        return self.generate_from_frequencies(frequencies)

    def generate_from_frequencies(self, frequencies, max_font_size=None):  # noqa: C901
        """Create a word_cloud from words and frequencies.

        Parameters
        ----------
        frequencies : dict from string to float
            A contains words and associated frequency.

        max_font_size : int
            Use this font-size instead of self.max_font_size

        Returns
        -------
        self

        """
        # make sure frequencies are sorted and normalized
        frequencies = sorted(frequencies.items(), key=itemgetter(1), reverse=True)


        if len(frequencies) <= 0:
            raise ValueError("We need at least 1 word to plot a word cloud, "
                             "got %d." % len(frequencies))
        frequencies = frequencies[:self.max_words]

        # largest entry will be 1
        max_frequency = float(frequencies[0][1])

        frequencies = [(word, freq / max_frequency)
                       for word, freq in frequencies]

        if self.random_state is not None:
            random_state = self.random_state
        else:
            random_state = Random()

        if self.mask is not None:
            boolean_mask = self._get_bolean_mask(self.mask)
            width = self.mask.shape[1]
            height = self.mask.shape[0]
        else:
            boolean_mask = None
            height, width = self.height, self.width
        occupancy = IntegralOccupancyMap(height, width, boolean_mask)

        # create image
        img_grey = Image.new("L", (width, height))
        draw = ImageDraw.Draw(img_grey)
        img_array = np.asarray(img_grey)
        font_sizes, positions, orientations, colors = [], [], [], []

        last_freq = 1.

        if max_font_size is None:
            # if not provided use default font_size
            max_font_size = self.max_font_size

        if max_font_size is None:
            # figure out a good font size by trying to draw with
            # just the first two words
            if len(frequencies) == 1:
                # we only have one word. We make it big!
                font_size = self.height
            else:
                self.generate_from_frequencies(dict(frequencies[:2]),
                                               max_font_size=self.height)
                # find font sizes
                sizes = [x[1] for x in self.layout_]
                try:
                    font_size = int(2 * sizes[0] * sizes[1]
                                    / (sizes[0] + sizes[1]))
                # quick fix for if self.layout_ contains less than 2 values
                # on very small images it can be empty
                except IndexError:
                    try:
                        font_size = sizes[0]
                    except IndexError:
                        raise ValueError(
                            "Couldn't find space to draw. Either the Canvas size"
                            " is too small or too much of the image is masked "
                            "out.")
        else:
            font_size = max_font_size

        # we set self.words_ here because we called generate_from_frequencies
        # above... hurray for good design?
        self.words_ = dict(frequencies)

        if self.repeat and len(frequencies) < self.max_words:
            # pad frequencies with repeating words.
            times_extend = int(np.ceil(self.max_words / len(frequencies))) - 1
            # get smallest frequency
            frequencies_org = list(frequencies)
            downweight = frequencies[-1][1]
            for i in range(times_extend):
                frequencies.extend([(word, freq * downweight ** (i + 1))
                                    for word, freq in frequencies_org])

        # start drawing grey image
        for word, freq in frequencies:
            # select the font size
            rs = self.relative_scaling
            if rs != 0:
                font_size = int(round((rs * (freq / float(last_freq))
                                       + (1 - rs)) * font_size))
            if random_state.random() < self.prefer_horizontal:
                orientation = None
            else:
                orientation = Image.ROTATE_90
            tried_other_orientation = False
            while True:
                # try to find a position
                font = ImageFont.truetype(self.font_path, font_size)
                # transpose font optionally
                transposed_font = ImageFont.TransposedFont(
                    font, orientation=orientation)
                # get size of resulting text
                box_size = draw.textsize(word, font=transposed_font)
                # find possible places using integral image:
                result = occupancy.sample_position_v2(box_size[0] + self.margin,
                                                   box_size[1] + self.margin,
                                                   random_state)

                if result is not None or font_size < self.min_font_size:
                    # either we found a place or font-size went too small
                    break
                # if we didn't find a place, make font smaller
                # but first try to rotate!
                if not tried_other_orientation and self.prefer_horizontal < 1:
                    orientation = (Image.ROTATE_90 if orientation is None else
                                   Image.ROTATE_90)
                    tried_other_orientation = True
                else:
                    font_size -= self.font_step
                    orientation = None

            if font_size < self.min_font_size:
                # we were unable to draw any more
                break

            x, y = np.array(result) + self.margin // 2

            # actually draw the text
            draw.text((y, x), word, fill="white", font=transposed_font)
            positions.append((x, y))
            orientations.append(orientation)
            font_sizes.append(font_size)
            colors.append(self.color_func(word, font_size=font_size,
                                          position=(x, y),
                                          orientation=orientation,
                                          random_state=random_state,
                                          font_path=self.font_path))
            # recompute integral image
            if self.mask is None:
                img_array = np.asarray(img_grey)
            else:
                img_array = np.asarray(img_grey) + boolean_mask
            # recompute bottom right
            # the order of the cumsum's is important for speed ?!
            occupancy.update(img_array, x, y)
            last_freq = freq

        self.layout_ = list(zip(frequencies, font_sizes, positions,
                               orientations, colors))
        return self


    def Color_Distribution(self, i):
        Hue_Spectrum = ['rgb(29, 122, 202)', 'rgb(0, 139, 182)', 'rgb(1, 146, 149)', 'rgb(0, 146, 107)',
                        'rgb(1, 142, 46)', 'rgb(78, 134, 0)', 'rgb(131, 123, 0)', 'rgb(161, 111, 0)',
                        'rgb(182, 94, 46)', 'rgb(197, 78, 110)', 'rgb(197, 69, 152)', 'rgb(180, 77, 186)',
                        'rgb(135, 98, 204)']
        l = len(Hue_Spectrum)
        c = 3
        return Hue_Spectrum[(i + c) % l]

    def GeoGenerate(self, words, A , Am ,cluster_points, PointsNum, max_font_size=None):
        '''
        :param words: Preprocessed data sets
        :param A: the area of the rectangle bounding region M
        :param Am: the area of M in pixel
        :param geomap : the region coordinate
        :param cluster_points : the points that labeled with the cluster index
        :param max_font_size: Max Font Size
        :return: Geo Word Cloud

        '''
        # Coverage error
        Measure1 = 0.0
        # Words not represent
        Measure2 = 0.0
        # difference between M and M'
        Measure3 = 0.0
        MM = np.zeros((800, 500))
        # Cluster center error
        Measure4 = []
        # the total frequency
        n = 0
        que = Q.PriorityQueue()
        min_freq = 999999999
        for i in range(0, words.shape[0], 1):
            que.put(WordCluster(words[i][0], words[i][1], words[i][2], words[i][3], words[i][4], words[i][5],words[i][6],words[i][7],words[i][8], i+1))
            n = n + words[i][3]
            if min_freq > words[i][3]:
                min_freq = words[i][3]

        # the number of the cluster
        CountWord  = que.qsize()

        '''
        #users can diy the fill word to complete the shape of the map
        
        if que.qsize() < self.max_words:
            for i in range(0, self.max_words - que.qsize(), 1):
                # users can diy fill words
                que.put(WordCluster('fill', self.width//2, self.height//2, (int)(min_freq * 0.5) , 1))
        '''

        frequencies = list()

        if self.random_state is not None:
            random_state = self.random_state
        else:
            random_state = Random()

        # set the mask
        if self.mask is not None:
            boolean_mask = self._get_bolean_mask(self.mask)
            width = self.mask.shape[1]
            height = self.mask.shape[0]
        else:
            boolean_mask = None
            height, width = self.height, self.width

        occupancy = IntegralOccupancyMap(height, width, boolean_mask)

        # create image
        img_grey = Image.new("L", (width, height))
        draw = ImageDraw.Draw(img_grey)
        img_array = np.asarray(img_grey)
        font_sizes, positions, orientations, colors = [], [], [], []

        last_freq = 1.

        if max_font_size is None:
            # if not provided use default font_size
            max_font_size = self.max_font_size

        curidx = 0

        # start drawing grey image
        while not que.empty():
            curword = que.get()
            freq = curword.num
            word = curword.word
            # the cluster center
            center_x = (int)(curword.x)
            center_y = (int)(curword.y)
            # the cluster range
            min_x = (int)(curword.min_x)
            max_x = (int)(curword.max_x)
            min_y = (int)(curword.min_y)
            max_y = (int)(curword.max_y)

            # select the font size
            rs = self.relative_scaling
            if rs != 0:
                font_size = int((cmath.sqrt((freq * Am)/n).real))

            # select the rotation
            if curword.direction == 1:
                orientation = None
            else:
                orientation = Image.ROTATE_90

            tried_other_orientation = False
            #Origin_orientation = orientation

            while True:
                # try to find a position
                font = ImageFont.truetype(self.font_path, font_size)
                # transpose font optionally
                transposed_font = ImageFont.TransposedFont(
                    font, orientation=orientation)
                # get size of resulting text
                box_size = draw.textsize(word, font=transposed_font)
                # find possible places using integral image:
                '''
                if freq < min_freq:
                    result = occupancy.sample_position_v2(box_size[0] + self.margin, box_size[1] + self.margin,
                                                          1, 1,self.width ,self.height )
                else:
                '''
                '''
                b_x = geomap[CurRegion][0]
                b_y = geomap[CurRegion][2]
                e_x = geomap[CurRegion][1]
                e_y = geomap[CurRegion][3]
                '''

                result = occupancy.sample_position_v2(box_size[0] + self.margin, box_size[1] + self.margin,
                                                     min_x, min_y, max_x, max_y,center_x,center_y)

                if result is not None or font_size < self.min_font_size:
                    # either we found a place or font-size went too small
                    break
                # if we didn't find a place, make font smaller
                # but first try to rotate
                if not tried_other_orientation :
                    orientation = (Image.ROTATE_90 if orientation is None else Image.ROTATE_270)
                    tried_other_orientation = True
                    continue
                # scale the word down
                else:
                    weight_aver = 0.5
                    penalty = 9999999999.9
                    scaling_factor = 1.0
                    # find the best placement
                    b_x = center_x - (box_size[0] + self.margin) // 2
                    b_y = center_y - (box_size[1] + self.margin) // 2
                    for s_f in np.arange(0.05,1.00,0.05):
                        tmp_penalty = weight_aver * (1 - s_f)
                        new_x = (int)(center_x - box_size[0] * s_f / 2)
                        new_y = (int)(center_y - box_size[1] * s_f / 2)
                        distance_error = cmath.sqrt((b_x - new_x ) * (b_x - new_x ) + (b_y - new_y) * (b_y - new_y))
                        tmp_penalty = tmp_penalty + (1 - weight_aver) * distance_error / cmath.sqrt(A)
                        if tmp_penalty <= penalty:
                            scaling_factor = s_f
                            penalty = tmp_penalty
                    freq = (int)(freq * scaling_factor)
                    que.put(WordCluster(word, center_x, center_y , freq, curword.direction, min_x, max_x, min_y, max_y, curword.idx))
                    break

            # can place
            if result is not None:
                x, y = np.array(result) + self.margin // 2
                # actually draw the text
                draw.text((y, x), word, fill="white", font=transposed_font)
                tmpmeasure = Hausdorff_Distance(y,x,box_size[0] + self.margin,box_size[1] + self.margin,curword.idx,cluster_points)
                tmp1, tmp2 = np.array(tmpmeasure)
                Measure1 = Measure1 + tmp1
                Measure2 = Measure2 + tmp2
                Measure4.append(Hausdorff_DistanceVer2(y,x,box_size[0] + self.margin,box_size[1] + self.margin,center_x,center_y))
                Fill_Map(MM, y, x,box_size[0] + self.margin, box_size[1] + self.margin)
                # append attributes
                frequencies.append((word, freq))
                positions.append((x, y))
                orientations.append(orientation)
                font_sizes.append(font_size)
                '''
                colors.append(self.color_func(word, font_size=font_size,
                                              position=(x, y),
                                              orientation=orientation,
                                              random_state=random_state,
                                              font_path=self.font_path))
                '''
                colors.append(self.Color_Distribution(curidx))
                curidx = curidx + 1
                # print(colors)
                # recompute integral image
                if self.mask is None:
                    img_array = np.asarray(img_grey)
                else:
                    img_array = np.asarray(img_grey) + boolean_mask
                # recompute bottom right
                # the order of the cumsum's is important for speed ?!
                occupancy.update(img_array, x, y)
                last_freq = freq

        # layout
        self.layout_ = list(zip(frequencies, font_sizes, positions,
                                orientations, colors))


        # Three Measures
        Measure1 = (Measure1/PointsNum).real
        Measure2 = (PointsNum - Measure2)/PointsNum
        Measure3 = (162384 - len(MM[MM == 1]))/162384
        Measure4.sort()

        print ("Measure1:")
        print (Measure1)
        print (Measure1/680)
        print (Measure1/440)
        print ("Measure2:")
        print (Measure2)
        print("Measure3:")
        print(Measure3)
        print("Measure4:")
        print(Measure4[len(Measure4)//2])
        print(Measure4)
        return self

    def process_text(self, text):
        """Splits a long text into words, eliminates the stopwords.

        Parameters
        ----------
        text : string
            The text to be processed.

        Returns
        -------
        words : dict (string, int)
            Word tokens with associated frequency.

        ..versionchanged:: 1.2.2
            Changed return type from list of tuples to dict.

        Notes
        -----
        There are better ways to do word tokenization, but I don't want to
        include all those things.
        """

        stopwords = set([i.lower() for i in self.stopwords])

        flags = (re.UNICODE if sys.version < '3' and type(text) is unicode  # noqa: F821
                 else 0)
        regexp = self.regexp if self.regexp is not None else r"\w[\w']+"

        words = re.findall(regexp, text, flags)
        # remove stopwords
        words = [word for word in words if word.lower() not in stopwords]
        # remove 's
        words = [word[:-2] if word.lower().endswith("'s") else word
                 for word in words]
        # remove numbers
        words = [word for word in words if not word.isdigit()]

        if self.collocations:
            word_counts = unigrams_and_bigrams(words, self.normalize_plurals)
        else:
            word_counts, _ = process_tokens(words, self.normalize_plurals)

        return word_counts

    def generate_from_text(self, text):
        """Generate wordcloud from text.

        The input "text" is expected to be a natural text. If you pass a sorted
        list of words, words will appear in your output twice. To remove this
        duplication, set ``collocations=False``.

        Calls process_text and generate_from_frequencies.

        ..versionchanged:: 1.2.2
            Argument of generate_from_frequencies() is not return of
            process_text() any more.

        Returns
        -------
        self
        """
        words = self.process_text(text)
        self.generate_from_frequencies(words)
        return self

    def generate(self, text):
        """Generate wordcloud from text.

        The input "text" is expected to be a natural text. If you pass a sorted
        list of words, words will appear in your output twice. To remove this
        duplication, set ``collocations=False``.

        Alias to generate_from_text.

        Calls process_text and generate_from_frequencies.

        Returns
        -------
        self
        """
        return self.generate_from_text(text)



    def _check_generated(self):
        """Check if ``layout_`` was computed, otherwise raise error."""
        if not hasattr(self, "layout_"):
            raise ValueError("WordCloud has not been calculated, call generate"
                             " first.")

    def to_image(self):
        self._check_generated()
        if self.mask is not None:
            width = self.mask.shape[1]
            height = self.mask.shape[0]
        else:
            height, width = self.height, self.width

        img = Image.new(self.mode, (int(width * self.scale),
                                    int(height * self.scale)),
                        self.background_color)
        draw = ImageDraw.Draw(img)
        for (word, count), font_size, position, orientation, color in self.layout_:
            font = ImageFont.truetype(self.font_path,
                                      int(font_size * self.scale))
            transposed_font = ImageFont.TransposedFont(
                font, orientation=orientation)
            pos = (int(position[1] * self.scale),
                   int(position[0] * self.scale))
            draw.text(pos, word, fill=color, font=transposed_font)

        return self._draw_contour(img=img)

    def recolor(self, random_state=None, color_func=None, colormap=None):
        """Recolor existing layout.

        Applying a new coloring is much faster than generating the whole
        wordcloud.

        Parameters
        ----------
        random_state : RandomState, int, or None, default=None
            If not None, a fixed random state is used. If an int is given, this
            is used as seed for a random.Random state.

        color_func : function or None, default=None
            Function to generate new color from word count, font size, position
            and orientation.  If None, self.color_func is used.

        colormap : string or matplotlib colormap, default=None
            Use this colormap to generate new colors. Ignored if color_func
            is specified. If None, self.color_func (or self.color_map) is used.

        Returns
        -------
        self
        """
        if isinstance(random_state, int):
            random_state = Random(random_state)
        self._check_generated()

        if color_func is None:
            if colormap is None:
                color_func = self.color_func
            else:
                color_func = colormap_color_func(colormap)
        self.layout_ = [(word_freq, font_size, position, orientation,
                         color_func(word=word_freq[0], font_size=font_size,
                                    position=position, orientation=orientation,
                                    random_state=random_state,
                                    font_path=self.font_path))
                        for word_freq, font_size, position, orientation, _
                        in self.layout_]
        return self

    def to_file(self, filename):
        """Export to image file.

        Parameters
        ----------
        filename : string
            Location to write to.

        Returns
        -------
        self
        """

        img = self.to_image()
        img.save(filename, optimize=True)
        return self

    def to_array(self):
        """Convert to numpy array.

        Returns
        -------
        image : nd-array size (width, height, 3)
            Word cloud image as numpy matrix.
        """
        return np.array(self.to_image())

    def __array__(self):
        """Convert to numpy array.

        Returns
        -------
        image : nd-array size (width, height, 3)
            Word cloud image as numpy matrix.
        """
        return self.to_array()

    def to_html(self):
        raise NotImplementedError("FIXME!!!")

    def _get_bolean_mask(self, mask):
        """Cast to two dimensional boolean mask."""
        if mask.dtype.kind == 'f':
            warnings.warn("mask image should be unsigned byte between 0"
                          " and 255. Got a float array")
        if mask.ndim == 2:
            boolean_mask = mask == 255
        elif mask.ndim == 3:
            # if all channels are white, mask out
            boolean_mask = np.all(mask[:, :, :3] == 255, axis=-1)
        else:
            raise ValueError("Got mask of invalid shape: %s" % str(mask.shape))
        return boolean_mask

    def _draw_contour(self, img):
        """Draw mask contour on a pillow image."""
        if self.mask is None or self.contour_width == 0:
            return img

        mask = self._get_bolean_mask(self.mask) * 255
        contour = Image.fromarray(mask.astype(np.uint8))
        contour = contour.resize(img.size)
        contour = contour.filter(ImageFilter.FIND_EDGES)
        contour = np.array(contour)

        # make sure borders are not drawn before changing width
        contour[[0, -1], :] = 0
        contour[:, [0, -1]] = 0

        # use gaussian to change width, divide by 10 to give more resolution
        radius = self.contour_width / 10
        contour = Image.fromarray(contour)
        contour = contour.filter(ImageFilter.GaussianBlur(radius=radius))
        contour = np.array(contour) > 0
        contour = np.dstack((contour, contour, contour))

        # color the contour
        ret = np.array(img) * np.invert(contour)
        if self.contour_color != 'black':
            color = Image.new(img.mode, img.size, self.contour_color)
            ret += np.array(color) * contour

        return Image.fromarray(ret)