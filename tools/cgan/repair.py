import sys

converted_feats = sys.argv[1]
original_script = sys.argv[2]

feat = {}
with open(converted_feats) as f:
    for l in f:
        name = l.strip().split('/')[-1].split('.')[0]
        feat[name] = l.strip()

with open(original_script) as f:
    for l in f:
        dir, txt = l.strip().split(' ', 1)
        name = dir.split('/')[-1].split('.')[0]
        if name in feat:
            print('{} {}'.format(feat[name], txt))
        else:
            print(l.strip())
