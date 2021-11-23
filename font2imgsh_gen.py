import os
import sys

import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['imgs2imgs', 'font2imgs', 'font2font', 'fonts2imgs'], required=True,
                    help='generate mode.\n'
                         'use --src_imgs and --dst_imgs for imgs2imgs mode.\n'
                         'use --src_font and --dst_imgs for font2imgs mode.\n'
                         'use --src_font and --dst_font for font2font mode.\n'
                         'use --src_fonts_dir and --dst_imgs for fonts2imgs mode.\n'
                         'No imgs2font mode.'
                    )
parser.add_argument('--src_font', type=str, default=None, help='path of the source font')
parser.add_argument('--src_fonts_dir', type=str, default=None, help='path of the source fonts')
parser.add_argument('--src_imgs', type=str, default=None, help='path of the source imgs')
parser.add_argument('--dst_fonts_dir', type=str, required=True, help='path of the target font')
parser.add_argument('--dst_imgs', type=str, default=None, help='path of the target imgs')

parser.add_argument('--filter', default=False, action='store_true', help='filter recurring characters')
parser.add_argument('--charset', type=str, default='CN',
                    help='charset, can be either: CN, JP, KR or a one line file. ONLY VALID IN font2font mode.')
parser.add_argument('--shuffle', default=False, action='store_true', help='shuffle a charset before processings')
parser.add_argument('--char_size', type=int, default=256, help='character size')
parser.add_argument('--canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', type=int, default=0, help='x offset')
parser.add_argument('--y_offset', type=int, default=0, help='y_offset')
parser.add_argument('--sample_count', type=int, default=5000, help='number of characters to draw')
parser.add_argument('--sample_dir', type=str, default='sample_dir', help='directory to save examples')
parser.add_argument('--label', type=int, default=0, help='label as the prefix of examples')

args = parser.parse_args()

if __name__ == '__main__':
    commands = []
    commands.append(f'--mode {args.mode}')
    if args.src_font is not None:
        commands.append(f'--src_font {args.src_font}')
    if args.src_fonts_dir is not None:
        commands.append(f'--src_fonts_dir {args.src_fonts_dir}')
    if args.src_imgs is not None:
        commands.append(f'--src_imgs {args.src_imgs}')
    if args.dst_imgs is not None:
        commands.append(f'--dst_imgs {args.dst_imgs}')
    if args.filter:
        commands.append(f'--filter')
    if args.charset != 'CN':
        commands.append(f'--charset {args.charset}')
    if args.shuffle:
        commands.append(f'--shuffle')
    if args.canvas_size != 256:
        commands.append(f'--canvas_size {args.canvas_size}')
    if args.x_offset != 0:
        commands.append(f'--x_offset {args.x_offset}')
    if args.y_offset != 0:
        commands.append(f'--y_offset {args.y_offset}')
    if args.sample_count != 5000:
        commands.append(f'--sample_count {args.sample_count}')
    if args.sample_dir != 'sample_dir':
        commands.append(f'--sample_dir {args.sample_dir}')

    commands = ' '.join(commands)

    with open('font2img.sh', 'w') as fout:
        print('#! /bin/sh\n', file=fout)
        for i, file in enumerate(glob.glob(os.path.join(args.dst_fonts_dir, '*.*tf'))):
            print(f'python -B font2img.py {commands} --dst_font={file} --label={args.label+i}', file=fout)
