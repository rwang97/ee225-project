from pathlib import Path
import librosa
import numpy as np
import random
import pyrubberband as pyrb
import soundfile as sf
import shutil

root = Path('librispeech_finetuning/1h/0')
output_root = Path('librispeech_finetuning/1h/0_augment')

paths = []
def find_flac(path):
    for file in path.iterdir():
        if file.is_dir():
            find_flac(file)
        elif file.suffix == '.flac':
            paths.append(file)

transcript_paths = []
def find_transcript(path):
    for file in path.iterdir():
        if file.is_dir():
            find_transcript(file)
        elif file.suffix == '.txt':
            transcript_paths.append(file)

# pitch
def manipulate_pitch(data, sr, pitch_factor):
    return pyrb.pitch_shift(data, sr, pitch_factor)
    # return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

# speed
def manipulate_speed(data, sr, speed_factor):
    return pyrb.time_stretch(data, sr, speed_factor)
    # return librosa.effects.time_stretch(data, speed_factor)

# noise
def manipulate_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

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

def copy_transcript(root, output_root):
    for path in transcript_paths:
        output_path = output_root / find_save_path(path)
        shutil.copy(path, output_path)


find_flac(root)
find_transcript(root)
for path in paths:
    output_path = output_root / find_save_path(path)
    y, sr = sf.read(str(path))
    p_noise = 1
    if (random.uniform(0,1) > 1 - p_noise):
        noise_factor = random.uniform(0.00001, 0.00003)
        y = manipulate_noise(y, noise_factor)

    p_speed = 1
    if (random.uniform(0,1) > 1 - p_speed):
        speed_factor = random.uniform(0.75, 1.25)

        # p_slow = 0.5
        # if (random.uniform(0,1) > 1 - p_slow):
        #     speed_factor = random.uniform(0.5, 0.9)
        # else:
        #     speed_factor = random.uniform(1.1, 1.5)
        y = manipulate_speed(y, sr, speed_factor)

    p_pitch = 1
    if (random.uniform(0,1) > 1 - p_pitch):
        pitch_factor = random.randint(-3, 3)

        # p_high = 0.5
        # if (random.uniform(0,1) > 1 - p_high):
        #     pitch_factor = random.randint(1, 4)
        # else:
        #     pitch_factor = random.randint(-4, -1)

        y = manipulate_pitch(y, sr, pitch_factor)

    print(path, noise_factor, speed_factor, pitch_factor)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    sf.write(output_path, y, sr)


copy_transcript(root, output_root)
