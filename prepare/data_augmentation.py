import os
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.io import wavfile
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, SpecFrequencyMask, TimeMask, AirAbsorption

def apply_augmentation(wav, sr, augmentation):
    augmented_wav = augmentation(samples=wav, sample_rate=sr)
    return augmented_wav

def resample_wave(wav_in, wav_out, sample_rate, augmentation_prob):
    wav, sr = librosa.load(wav_in, sr=sample_rate)
    wav = wav / np.abs(wav).max() * 0.6
    wav = wav / max(0.01, np.max(np.abs(wav))) * 32767 * 0.6

    # 数据增强：加噪声、时间拉伸和压缩、随机频率掩蔽、随机时间遮蔽
    augmentation = Compose([
        AddGaussianNoise(p=augmentation_prob['gaussian_noise']),
        TimeStretch(min_rate=augmentation_prob['time_stretch'][0], max_rate=augmentation_prob['time_stretch'][1], p=augmentation_prob['time_stretch_prob']),
        PitchShift(min_semitones=augmentation_prob['pitch_shift'][0], max_semitones=augmentation_prob['pitch_shift'][1], p=augmentation_prob['pitch_shift_prob']),
        AirAbsorption(
            min_distance=5.0,
            max_distance=15.0,
            p=0.4,
        ),
        TimeMask(min_band_part=augmentation_prob['time_mask'][0], max_band_part=augmentation_prob['time_mask'][1], p=augmentation_prob['time_mask_prob']),
    ])

    augmented_wav = wav

    if np.random.rand() < augmentation_prob['overall_prob']:
        augmented_wav = apply_augmentation(wav, sr, augmentation)
        applied_augmentations = []
        
        # if np.random.rand() < augmentation_prob['gaussian_noise']:
        #     applied_augmentations.append('gaussian_noise')
        # if np.random.rand() < augmentation_prob['time_stretch_prob']:
        #     applied_augmentations.append('time_stretch')
        # if np.random.rand() < augmentation_prob['pitch_shift_prob']:
        #     applied_augmentations.append('pitch_shift')
        # if np.random.rand() < augmentation_prob['frequency_mask_prob']:
        #     applied_augmentations.append('frequency_mask')
        # if np.random.rand() < augmentation_prob['time_mask_prob']:
        #     applied_augmentations.append('time_mask')

        
        wavfile.write(f"{wav_out}_augmentation.wav", sample_rate, augmented_wav.astype(np.int16))
    else:

        wavfile.write(f"{wav_out}_original.wav", sample_rate, wav.astype(np.int16))

def process_file(file, wavPath, spks, outPath, sr, augmentation_prob):
    if file.endswith(".wav"):
        file = file[:-4]
        resample_wave(f"{wavPath}/{spks}/{file}.wav", f"{outPath}/{spks}/{file}", sr, augmentation_prob)

def process_files_with_thread_pool(wavPath, spks, outPath, sr, thread_num, augmentation_prob):
    files = [f for f in os.listdir(f"./{wavPath}/{spks}") if f.endswith(".wav")]

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = {executor.submit(process_file, file, wavPath, spks, outPath, sr, augmentation_prob): file for file in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing {sr} {spks}'):
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-o", "--out", help="out", dest="out", required=True)
    parser.add_argument("-s", "--sr", help="sample rate", dest="sr", type=int, required=True)
    parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all CPU cores", dest="thread_count", type=int, default=1)
    parser.add_argument("--overall_prob", help="overall probability of applying augmentation", dest="overall_prob", type=float, default=0.4)
    parser.add_argument("--gaussian_noise_prob", help="probability of adding Gaussian noise", dest="gaussian_noise_prob", type=float, default=0.4)
    parser.add_argument("--time_stretch_prob", help="probability of time stretching", dest="time_stretch_prob", type=float, default=0.4)
    parser.add_argument("--pitch_shift_prob", help="probability of pitch shifting", dest="pitch_shift_prob", type=float, default=0.4)
    parser.add_argument("--frequency_mask_prob", help="probability of frequency masking", dest="frequency_mask_prob", type=float, default=0.4)
    parser.add_argument("--time_mask_prob", help="probability of time masking", dest="time_mask_prob", type=float, default=0.4)

    args = parser.parse_args()
    print(args.wav)
    print(args.out)
    print(args.sr)

    os.makedirs(args.out, exist_ok=True)
    wavPath = args.wav
    outPath = args.out

    assert args.sr == 16000 or args.sr == 32000
    print('assert OK')

    augmentation_prob = {
        'overall_prob': args.overall_prob,
        'gaussian_noise': args.gaussian_noise_prob,
        'time_stretch_prob': args.time_stretch_prob,
        'pitch_shift_prob': args.pitch_shift_prob,
        'frequency_mask_prob': args.frequency_mask_prob,
        'time_mask_prob': args.time_mask_prob,
        'time_stretch': (0.8, 1.2),
        'pitch_shift': (-2, 2),
        'frequency_mask': (0.01, 0.03),
        'time_mask': (0.01, 0.03),
        'min_frequency_mask': 0.01,
        'max_frequency_mask': 0.02,
    }

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{outPath}/{spks}", exist_ok=True)
            if args.thread_count == 0:
                process_num = os.cpu_count() // 2 + 1
            else:
                process_num = args.thread_count
            process_files_with_thread_pool(wavPath, spks, outPath, args.sr, process_num, augmentation_prob)
