import yt_dlp
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

@contextmanager
def temporary_directory() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

def extract_audio(url: str) -> bytes:
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': '%(id)s.%(ext)s',
    }

    with temporary_directory() as temp_dir:
        ydl_opts['outtmpl'] = str(temp_dir / ydl_opts['outtmpl'])

        print("Extracting audio from YouTube video...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
        
        audio_file = Path(filename).with_suffix('.mp3')
        print(f"Audio extracted: {audio_file.name}")

        with audio_file.open('rb') as f:
            audio_bytes = f.read()

    print("Audio extraction completed successfully.")
    return audio_bytes