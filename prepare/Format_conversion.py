import os
import argparse
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, wait, as_completed

from tqdm import tqdm

def convert_audio_to_wav(input_path, output_folder):
    input_folder, file_name = os.path.split(input_path)
    output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.wav")

    # Load the audio file
    audio = AudioSegment.from_file(input_path)

    # Export the audio to WAV format
    audio.export(output_path, format="wav")

def process_files_with_thread_pool(input_folder, output_folder, thread_num=None):
    files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.lower().endswith(('.mp3', '.ogg', '.flac', '.wav', '.aac'))]

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = {executor.submit(convert_audio_to_wav, file, output_folder): file for file in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc='Converting to WAV'):
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", help="Input folder containing audio files", dest="input_folder", required=True)
    parser.add_argument("-o", "--output_folder", help="Output folder for WAV files", dest="output_folder", required=True)
    parser.add_argument("-t", "--thread_count", help="Thread count to process, set 0 to use all CPU cores", dest="thread_count", type=int, default=1)

    args = parser.parse_args()
    print(args.input_folder)
    print(args.output_folder)

    os.makedirs(args.output_folder, exist_ok=True)

    process_files_with_thread_pool(args.input_folder, args.output_folder, args.thread_count)
