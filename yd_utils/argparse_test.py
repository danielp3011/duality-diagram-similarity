import argparse

parser = argparse.ArgumentParser(description="Test")

parser.add_argument("-s", "--stuff", type=str)

args = parser.parse_args()

def print_stuff(stuff):
    print(stuff)

if __name__ == '__main__':
    print_stuff(args.stuff)

# usage description:
# python3 argparse_test.py --stuff "hi wie gehts"