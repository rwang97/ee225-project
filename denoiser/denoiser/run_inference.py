# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import os
import sys
import shutil
from pathlib import Path

import torch
import torchaudio

from .audio import Audioset, find_audio_files
from . import distrib, pretrained
from .demucs import DemucsStreamer

from .utils import LogProgress

logger = logging.getLogger(__name__)


def add_flags(parser):
    """
    Add the flags for the argument parser that are related to model loading and evaluation"
    """
    pretrained.add_model_flags(parser)
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--dry', type=float, default=0,
                        help='dry/wet knob coefficient. 0 is only input signal, 1 only denoised.')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--streaming', action="store_true",
                        help="true streaming evaluation for Demucs")


parser = argparse.ArgumentParser(
        'denoiser.enhance',
        description="Speech enhancement using Demucs - Generate enhanced files")
add_flags(parser)
parser.add_argument("--out_dir", type=str, default="enhanced",
                    help="directory putting enhanced wav files")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="more loggging")

group = parser.add_mutually_exclusive_group()
group.add_argument("--noisy_dir", type=str, default=None,
                   help="directory including noisy wav files")
group.add_argument("--noisy_json", type=str, default=None,
                   help="json file including noisy wav files")


def get_estimate(model, noisy, args):
    torch.set_num_threads(1)
    if args.streaming:
        streamer = DemucsStreamer(model, dry=args.dry)
        with torch.no_grad():
            estimate = torch.cat([
                streamer.feed(noisy[0]),
                streamer.flush()], dim=1)[None]
    else:
        with torch.no_grad():
            estimate = model(noisy)
            estimate = (1 - args.dry) * estimate + args.dry * noisy
    return estimate


def save_wavs(estimates, noisy_sigs, filenames, out_dir, sr=16_000):
    # Write result
    for estimate, noisy, filename in zip(estimates, noisy_sigs, filenames):
        filename = os.path.join(out_dir, os.path.basename(filename).rsplit(".", 1)[0])
        # write(noisy, filename + "_noisy.wav", sr=sr)
        write(estimate, filename + ".flac", sr=sr)


def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def get_dataset(args, sample_rate, channels):
    if hasattr(args, 'dset'):
        paths = args.dset
    else:
        paths = args
    if paths.noisy_json:
        with open(paths.noisy_json) as f:
            files = json.load(f)
    elif paths.noisy_dir:
        files = find_audio_files(paths.noisy_dir)
    else:
        logger.warning(
            "Small sample set was not provided by either noisy_dir or noisy_json. "
            "Skipping enhancement.")
        return None
    return Audioset(files, with_path=True,
                    sample_rate=sample_rate, channels=channels, convert=True)


def _estimate_and_save(model, noisy_signals, filenames, out_dir, args):
    estimate = get_estimate(model, noisy_signals, args)
    save_wavs(estimate, noisy_signals, filenames, out_dir, sr=model.sample_rate)


def enhance(args, model=None, local_out_dir=None):
    # Load model
    if not model:
        model = pretrained.get_model(args).to(args.device)
    model.eval()
    if local_out_dir:
        out_dir = local_out_dir
    else:
        out_dir = args.out_dir

    dset = get_dataset(args, model.sample_rate, model.chin)
    if dset is None:
        return
    loader = distrib.loader(dset, batch_size=1)

    if distrib.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    distrib.barrier()

    with ProcessPoolExecutor(args.num_workers) as pool:
        iterator = LogProgress(logger, loader, name="Generate enhanced files")
        pendings = []
        for data in iterator:
            # Get batch data
            noisy_signals, filenames = data
            noisy_signals = noisy_signals.to(args.device)
            if args.device == 'cpu' and args.num_workers > 1:
                pendings.append(
                    pool.submit(_estimate_and_save,
                                model, noisy_signals, filenames, out_dir, args))
            else:
                # Forward
                estimate = get_estimate(model, noisy_signals, args)
                save_wavs(estimate, noisy_signals, filenames, out_dir, sr=model.sample_rate)

        if pendings:
            print('Waiting for pending jobs...')
            for pending in LogProgress(logger, pendings, updates=5, name="Generate enhanced files"):
                pending.result()

def check_child(path):
    for file in path.iterdir():
        if file.is_file():
            return True
        else:
            return False
    
def find_flac(path):
    for file in path.iterdir():
        if file.is_dir():
            if check_child(file):
                paths.append(file)
            else:
                find_flac(file)

def find_save_path(path, last_name='0'):
    path = str(path).split('/')
    save_path = []
    for child in reversed(path):
        if child == last_name:
            break
        save_path.append(child)
    
    save_path = reversed(save_path)
    save_path = '/'.join(save_path)

    return save_path

def find_transcript(path):
    for file in path.iterdir():
        if file.is_dir():
            find_transcript(file)
        elif file.suffix == '.txt':
            transcript_paths.append(file)

def copy_transcript(root, output_root):
    for path in transcript_paths:
        output_path = output_root / find_save_path(path)
        shutil.copy(path, output_path)

if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    paths = []
    transcript_paths = []

    input_dir = Path(args.noisy_dir)
    find_flac(input_dir)
    find_transcript(input_dir)

    print(paths)
    for path in paths:
        args.noisy_dir = path
        output_dir = Path(args.out_dir) / find_save_path(path)
        output_dir.mkdir(exist_ok=True, parents=True)
        enhance(args, local_out_dir=str(output_dir))
    
    copy_transcript(input_dir, Path(args.out_dir))
