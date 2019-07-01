

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--example', default=1, type=int)
parser.add_argument('--hello', default=1, type=int)
args = parser.parse_known_args()
print(args)