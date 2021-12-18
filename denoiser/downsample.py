import librosa
from pathlib import Path
import soundfile as sf


root = Path('dataset/train')
output_root = Path('dataset/train_16000')
paths = []
def find_path(root):
    for file in root.iterdir():
        if file.is_dir():
            find_path(file)
        elif file.suffix == '.wav':
            paths.append(file)

def find_save_path(path, last_name='train'):
    path = str(path).split('/')
    save_path = []
    for child in reversed(path):
        if child == last_name:
            break
        save_path.append(child)
    
    save_path = reversed(save_path)
    save_path = '/'.join(save_path)

    return save_path

find_path(root)

for path in paths:
    output_path = output_root / find_save_path(path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    y, s = librosa.load(str(path), sr=16000)
    sf.write(str(output_path), y, s)