"""
To add <sos> & <eos>
"""
import sys

prefix = '<sos>'
surfix = '<eos>'

txt_dir = sys.argv[1]

with open(txt_dir) as f:
    for l in f:
        dir, txt = l.strip().split(' ', 1)
        new_txt = prefix + ' '+ txt + ' ' + surfix
        print(' '.join([dir, new_txt]))

