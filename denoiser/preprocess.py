from __future__ import print_function
import subprocess
import soundfile as sf
import pyrubberband as pyrb
import argparse
from pathlib import Path
import librosa
import dtw
from audiotools import *
import shutil

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import librosa
import librosa.display
import time

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_dir", type=Path, default="/Users/rwang97/Desktop/denoiser/dataset/test/mask_ori")
    parser.add_argument("--input_dir2", type=Path, default="/Users/rwang97/Desktop/denoiser/dataset/test/no_mask_ori")
    parser.add_argument("--output_dir", type=Path, default="/Users/rwang97/Desktop/denoiser/dataset/test/mask")
    parser.add_argument("--output_dir2", type=Path, default="/Users/rwang97/Desktop/denoiser/dataset/test/no_mask")

    parser.add_argument("--prefix", type=str, default="MASK_")
    parser.add_argument("--prefix2", type=str, default="NO_")

    # parser.add_argument("--input_dir", type=Path, default="/Users/rwang97/Desktop/denoiser/dataset/eason/mask_ori")
    # parser.add_argument("--input_dir2", type=Path, default="/Users/rwang97/Desktop/denoiser/dataset/eason/no_mask_ori")
    # parser.add_argument("--output_dir", type=Path, default="/Users/rwang97/Desktop/denoiser/dataset/eason/mask")
    # parser.add_argument("--output_dir2", type=Path, default="/Users/rwang97/Desktop/denoiser/dataset/eason/no_mask")

    # parser.add_argument("--prefix", type=str, default="MASK4_")
    # parser.add_argument("--prefix2", type=str, default="NO4_")

    args = parser.parse_args()

    return args

def plot(x_1, x_2, fs, D, wp, hop_size):
    # x_1_chroma = librosa.feature.chroma_stft(y=x_1, sr=fs, tuning=0, norm=2,
    #                                         hop_length=hop_size, n_fft=n_fft)
    # x_2_chroma = librosa.feature.chroma_stft(y=x_2, sr=fs, tuning=0, norm=2,
    #                                         hop_length=hop_size, n_fft=n_fft)

    # print(x_1_chroma.shape)
    # print(x_2_chroma.shape)
    # D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine')
    # print(wp.shape)
    # print(wp)
    # wp_s = np.asarray(wp) * hop_size / fs

    # fig = plt.figure(figsize=(16, 8))

    # # Plot x_1
    # plt.subplot(2, 1, 1)
    # librosa.display.waveplot(x_1, sr=fs)
    # plt.title('Slower Version $X_1$')
    # ax1 = plt.gca()

    # # Plot x_2
    # plt.subplot(2, 1, 2)
    # librosa.display.waveplot(x_2, sr=fs)
    # plt.title('Slower Version $X_2$')
    # ax2 = plt.gca()

    # plt.tight_layout()

    # trans_figure = fig.transFigure.inverted()
    # lines = []
    # arrows = 30
    # points_idx = np.int16(np.round(np.linspace(0, wp.shape[0] - 1, arrows)))
    # # print(points_idx)
    # # print(len(points_idx))

    # # for tp1, tp2 in zip((wp[points_idx, 0]) * hop_size, (wp[points_idx, 1]) * hop_size):
    # for tp1, tp2 in wp[points_idx] * hop_size / fs:
    #     # get position on axis for a given index-pair
    #     coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
    #     coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))

    #     # draw a line
    #     line = matplotlib.lines.Line2D((coord1[0], coord2[0]),
    #                                 (coord1[1], coord2[1]),
    #                                 transform=fig.transFigure,
    #                                 color='r')
    #     lines.append(line)

    # fig.lines = lines
    # plt.tight_layout()
    # plt.show()


    wp_s = np.asarray(wp) * hop_size / fs

    fig = plt.figure(figsize=(16, 8))

    # Plot x_1
    plt.subplot(2, 1, 1)
    librosa.display.waveplot(x_1, sr=fs)
    plt.title('Audio With Mask', size=20)
    ax1 = plt.gca()

    # Plot x_2
    plt.subplot(2, 1, 2)
    librosa.display.waveplot(x_2, sr=fs)
    plt.title('Audio With No Mask', size=20)
    ax2 = plt.gca()

    plt.tight_layout()

    trans_figure = fig.transFigure.inverted()
    lines = []
    arrows = 30
    points_idx = np.int16(np.round(np.linspace(0, wp.shape[0] - 1, arrows)))
    # print(points_idx)
    # print(len(points_idx))

    # for tp1, tp2 in zip((wp[points_idx, 0]) * hop_size, (wp[points_idx, 1]) * hop_size):
    for tp1, tp2 in wp[points_idx] * hop_size / fs:
        # get position on axis for a given index-pair
        coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
        coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))

        # draw a line
        line = matplotlib.lines.Line2D((coord1[0], coord2[0]),
                                    (coord1[1], coord2[1]),
                                    transform=fig.transFigure,
                                    color='r')
        lines.append(line)

    fig.lines = lines
    plt.tight_layout()
    plt.savefig('alignment.pdf', dpi = 300, transparent = True, bbox_inches='tight')
    # plt.show()

def align(x_1, x_2, fs):

    hop_length = 1024
    X1 = get_mfcc_mod(x_1, fs, hop_length, 120, 0).T
    X2 = get_mfcc_mod(x_2, fs, hop_length, 120, 0).T
    tic = time.time()

    D, path = librosa.sequence.dtw(X=X1, Y=X2, metric='euclidean')

    # plot(x_1, x_2, fs, D, path, hop_length)
    # exit(0)

    path = np.flip(path, axis=0)
    toc = time.time()
    # print("Elapsed Time DTW: {:.3f} seconds".format(toc-tic))
    xres = stretch_audio(x_1, x_2, fs, path, hop_length)
    return xres


def get_length(audio):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)

def align_audio(audio1, audio2, output1, output2):
    # duration1 = get_length(audio1)
    # duration2 = get_length(audio2)

    # target_len = (duration1 + duration2) / 2
    # y1, sr1 = sf.read(audio1) # why gives two channels..
    y1, sr1 = librosa.load(audio1, sr=44100)
    # y1, sr1 = librosa.load(audio1, sr=16000)
    # y_stretch = pyrb.time_stretch(y1, sr1, rate=1, rbargs={"-D": "{}".format(target_len)})
    # sf.write(output1, y_stretch, sr1)

    # y2, sr2 = sf.read(audio2)
    y2, sr2 = librosa.load(audio2, sr=44100)
    # y2, sr2 = librosa.load(audio2)
    # y2, sr2 = librosa.load(audio2, sr=16000)
    # y_stretch = pyrb.time_stretch(y2, sr2, rate=1, rbargs={"-D": "{}".format(target_len)})
    # sf.write(output2, y_stretch, sr2)

    if len(y2) < len(y1):
        # y_stretch = align(y1, y2, sr1) # output will combine both audios

        y_stretch = align(y1, y2, sr1)[:,0]  # output is first dimension (x1 stretched), second dimension (same as x2)
        # print(y_stretch.shape, y1.shape, y2.shape)

        duration1 = librosa.get_duration(y_stretch, sr=sr1)
        duration2 = librosa.get_duration(y2, sr=sr2)
        target_len = (duration1 + duration2) / 2
        y_stretch = pyrb.time_stretch(y_stretch, sr1, rbargs={"-D": "{}".format(target_len)})
        y2 = pyrb.time_stretch(y2, sr2, rbargs={"-D": "{}".format(target_len)})

        if len(y2) < len(y_stretch):
            y_stretch = y_stretch[:len(y2)]
        elif len(y2) > len(y_stretch):
            y2 = y2[:len(y_stretch)]

        sf.write(output1, y_stretch, sr1)
        sf.write(output2, y2, sr2)

    else:
        # y_stretch = align(y2, y1, sr1)

        y_stretch = align(y2, y1, sr1)[:,0]

        duration1 = librosa.get_duration(y_stretch, sr=sr2)
        duration2 = librosa.get_duration(y1, sr=sr1)
        target_len = (duration1 + duration2) / 2

        y1 = pyrb.time_stretch(y1, sr1, rbargs={"-D": "{}".format(target_len)})
        y_stretch = pyrb.time_stretch(y_stretch, sr2, rbargs={"-D": "{}".format(target_len)})

        if len(y1) < len(y_stretch):
            y_stretch = y_stretch[:len(y1)]
        elif len(y1) > len(y_stretch):
            y1 = y1[:len(y_stretch)]

        sf.write(output2, y_stretch, sr2)
        sf.write(output1, y1, sr1)

def filter_audio_data(input_dir, output_dir_mask, output_dir_no):
    clear_DM_prefix = 'clear_DM_'
    clear_NO_prefix = 'clear_NO_'
    clear_TM_prefix = 'clear_TM_'

    conv_DM_prefix = 'conv_DM_'
    conv_NO_prefix = 'conv_NO_'
    conv_TM_prefix = 'conv_TM_'

    for i in range(1, 121):
        no_mask = input_dir / (clear_NO_prefix + str(i) + '.wav')
        no_mask_conv = input_dir / (conv_NO_prefix + str(i) + '.wav')

        output_no_mask = output_dir_no / ("NO_" + str(i) + '.wav')
        output_no_mask_conv = output_dir_no / ("NO_" + str(i + 120) + '.wav')
        output_no_mask_DM = output_dir_no / ("NO_" + str(i + 240) + '.wav')
        output_no_mask_DM_conv = output_dir_no / ("NO_" + str(i + 360) + '.wav')

        shutil.copy(no_mask, output_no_mask)
        shutil.copy(no_mask_conv, output_no_mask_conv)
        shutil.copy(no_mask, output_no_mask_DM)
        shutil.copy(no_mask_conv, output_no_mask_DM_conv)


        mask = input_dir / (clear_TM_prefix + str(i) + '.wav')
        mask_conv = input_dir / (conv_TM_prefix + str(i) + '.wav')
        mask_DM = input_dir / (clear_DM_prefix + str(i) + '.wav')
        mask_conv_DM = input_dir / (conv_DM_prefix + str(i) + '.wav')

        output_mask = output_dir_mask / ("MASK_" + str(i) + '.wav')
        output_mask_conv = output_dir_mask / ("MASK_" + str(i + 120) + '.wav')
        output_mask_DM = output_dir_mask / ("MASK_" + str(i + 240) + '.wav')
        output_mask_conv_DM = output_dir_mask / ("MASK_" + str(i + 360) + '.wav')

        shutil.copy(mask, output_mask)
        shutil.copy(mask_conv, output_mask_conv)
        shutil.copy(mask_DM, output_mask_DM)
        shutil.copy(mask_conv_DM, output_mask_conv_DM)

def loop_range(start, input_dir=None, output_dir_no=None, output_dir_mask=None, prefix_mask='cloth_mask_no_filter_', prefix_no='no_mask_', missing_num=0):
    # print(missing_num)
    start = start - missing_num
    cur_missing_num = 0

    for i in range(1, 161):
        no_mask = input_dir / (prefix_no + str(i) + '.wav')
        mask = input_dir / (prefix_mask + str(i) + '.wav')

        if not no_mask.exists():
            print(no_mask, " does not exist")
            cur_missing_num += 1
            continue

        if not mask.exists():
            print(mask, " does not exist")
            cur_missing_num += 1
        else:
            output_no_mask = output_dir_no / ("NO2_" + str(start) + '.wav')
            output_mask = output_dir_mask / ("MASK2_" + str(start) + '.wav')

            shutil.copy(no_mask, output_no_mask)
            shutil.copy(mask, output_mask)
            start += 1
    return cur_missing_num + missing_num


def filter_audio_data2(input_dir, output_dir_mask, output_dir_no):
    no_mask_prefix = 'no_mask_'
    cloth_filter_prefix = 'cloth_mask_no_filter_'
    cloth_prefix = 'cloth_mask_with_filter_'
    surgical_prefix = 'surgical_mask_'
    visible_prefix = 'visible_mouth_mask_'

    missing_num = 0

    missing_num = loop_range((1), input_dir, output_dir_no, output_dir_mask, prefix_mask=cloth_filter_prefix, prefix_no=no_mask_prefix, missing_num=missing_num)
    missing_num = loop_range((160+1), input_dir, output_dir_no, output_dir_mask, prefix_mask=cloth_prefix, prefix_no=no_mask_prefix, missing_num=missing_num)
    missing_num = loop_range((320+1), input_dir, output_dir_no, output_dir_mask, prefix_mask=surgical_prefix, prefix_no=no_mask_prefix, missing_num=missing_num)
    missing_num = loop_range((480+1), input_dir, output_dir_no, output_dir_mask, prefix_mask=visible_prefix, prefix_no=no_mask_prefix, missing_num=missing_num)

    return missing_num

def extract_audio(input_video, output_audio):
    import subprocess
    command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(input_video, output_audio)
    subprocess.call(command, shell=True)


def extract_audio_from_video(input_dir):
    output_dir = input_dir / ('_output')
    output_dir.mkdir(exist_ok=True, parents=True)

    for child in input_dir.iterdir():
        if child.is_dir():
            
            for file in child.iterdir():
                if file.suffix == ".mp4":
                    output_audio = output_dir / (file.stem + '.wav')
                    extract_audio(str(file), str(output_audio))


if __name__ == "__main__":
    # extract_audio_from_video(Path('raw_data/face'))

    args = parse_args()
    args.input_dir.mkdir(exist_ok=True, parents=True)
    args.input_dir2.mkdir(exist_ok=True, parents=True)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    args.output_dir2.mkdir(exist_ok=True, parents=True)

    filter_audio_data(Path('raw_data/effects/60db_audio'), args.input_dir, args.input_dir2)

    for i in range(1, 481):

        audio1 = args.input_dir / (args.prefix + str(i) + '.wav')
        audio2 = args.input_dir2 / (args.prefix2 + str(i) + '.wav')
        print(audio1, audio2)
        output1 = args.output_dir / (args.prefix + str(i) + '.wav')
        output2 = args.output_dir2 / (args.prefix2 + str(i) + '.wav')

        align_audio(audio1, audio2, output1, output2)

    print("========== start second dataset =============")

    missing_num = filter_audio_data2(Path('raw_data/face/audio'), args.input_dir, args.input_dir2)
    print(missing_num)
    args.prefix = 'MASK2_'
    args.prefix2 = 'NO2_'

    for i in range(1, 641 - missing_num):
    # for i in range(1, 11):

        audio1 = args.input_dir / (args.prefix + str(i) + '.wav')
        audio2 = args.input_dir2 / (args.prefix2 + str(i) + '.wav')
        print(audio1, audio2)
        output1 = args.output_dir / (args.prefix + str(i) + '.wav')
        output2 = args.output_dir2 / (args.prefix2 + str(i) + '.wav')

        align_audio(audio1, audio2, output1, output2)


    # filter russell audio
    # for i in range(180, 181):

    #     audio1 = args.input_dir / (args.prefix + str(i) + '.wav')
    #     audio2 = args.input_dir2 / (args.prefix2 + str(i) + '.wav')
    #     print(audio1, audio2)
    #     output1 = args.output_dir / (args.prefix + str(i) + '.wav')
    #     output2 = args.output_dir2 / (args.prefix2 + str(i) + '.wav')

    #     align_audio(audio1, audio2, output1, output2)


    # for i in range(1, 151):
    #     audio1 = args.input_dir / ('arctic_a{0:04d}'.format(i) + '.wav')
    #     audio2 = args.input_dir2 / ('arctic_a{0:04d}'.format(i) + '.wav')

    #     print(audio1, audio2)
    #     output1 = args.output_dir / (args.prefix + str(i) + '.wav')
    #     output2 = args.output_dir2 / (args.prefix2 + str(i) + '.wav')

    #     align_audio(audio1, audio2, output1, output2)