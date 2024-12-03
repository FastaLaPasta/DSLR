import sys
import os
from describe import treat_data
from histogram import disp_histogram


def main():
    if len(sys.argv) == 2:
        if os.path.isfile(sys.argv[1]):
            print(treat_data(sys.argv[1]))
            disp_histogram(sys.argv[1])
        else:
            raise FileExistsError(f'FileExistsError: \
Wrong File/Path for : {sys.argv[1]}')
    else:
        raise ValueError(f'ValueError: \
Need 2 arguments, current number of arguments: {len(sys.argv)}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
