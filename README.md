# ee225-project

This project investigates two data augmentation techniques to improve wav2vec2 performance in the low data regime. First is standard data augmentation in speech, including speed adjustment, pitch adjustment and random noise. The second is to simulate speech sound so that as if the speaker is wearing a face mask (masked speech augmentation). We use augmented data to fine-tune wav2vec2 and have a lower word error rate.


## standard data augmentation
Library requirements: rubberband, soundfile, librosa

The implementation is in [./standard_aug/augment.py](standard_aug/augment.py). You can change the `root` and `output_root` parameters for input and output path respectively and then run:
```
python augment.py
```

## masked speech generation
For this augmentation, we use a denoiser network by facebook, their codebase is [here](https://github.com/facebookresearch/denoiser). 

Library requirements: everything mentioned in their original repo plus rubberband if you want to run speech alignment.

To run speech alignment, please check `preprocess.py` for more details.

Please prepare your own dataset, and place it somewhere in the repo, and then change [make_train.sh]() to point to the desired input and output path. After that, run `sh make_train.sh` to generate json files under `egs/train/tr`. Additionally, please modify `conf/dset/train.yaml` file to point to the right training/validation/test path. The last thing you need to make sure in `conf/config.yaml`, the `dset` parameter is set correctly. After everything is completed, simply run:
```
python train.py
```

The output and models will be saved to `outputs`. To denoise the speech (i.e, generate masked speech), run:
```
python -m denoiser.enhance --model_path=<path to the model> --noisy_dir=<path to the dir with the noisy files> --out_dir=<path to store enhanced files>
```


## Fine-tune wav2vec2
We use the official wav2vec repo, and their codebase is [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec). 

Library requirements: everything mentioned in their original repo

After all required libraries are installed, here are some example procedures for preparing the data:
1. Download Libri-Light data (available [here](https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md))
2. 
```
python wav2vec_manifest.py /home/rwang97/fairseq/librispeech_finetuning/1h/0 --dest libri_light --ext flac --valid-percent 0
```
3. 
```
python libri_labels.py libri_light/train.tsv --output-dir libri_light --output-name train
```
4. Repeat above for preparing validation data
5. 
```
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt to download dictionary to place in libri_light
```
6. Download wav2vec pretrained model:
```
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt
```
7. Make sure fairseq/examples/wav2vec/config/finetuning/base_10m.yaml (you can use other yaml) is correct, especially check `valid_subset` to be the correct name
8. Run fine-tuning
```
fairseq-hydra-train task.data=<path to libri_light> model.w2v_path=<path to wav2vec_small.pt> distributed_training.distributed_world_size=1 +optimization.update_freq='[24]' --config-dir /home/rwang97/fairseq/examples/wav2vec/config/finetuning --config-name base_10m
```
9. Evaluation
```
# download langauge model and lexicon
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/librispeech_lexicon.lst
wget https://openslr.magicdatatech.com/resources/11/4-gram.arpa.gz

python fairseq/examples/speech_recognition/infer.py <path to libri_light> --task audio_finetuning \
      --nbest 1 --path <fine-tuned model path> --gen-subset test --results-path <path to where you want to save results> \
      --w2l-decoder kenlm --lm-model <path to 4-gram.arpa.gz> \
      --lm-weight 3.86 --word-score -0.1.18 --sil-weight 0 \
      --criterion ctc --labels ltr --max-tokens 4000000 --post-process letter --beam=1500 \
      --lexicon <path to librispeech_lexicon.lst> \
```


Note: There is a weird bug when running `infer.py`, you can probably fix by adding a line `task_cfg.labels = 'ltr'` to `fairseq/fairseq/tasks/audio_finetuning.py`