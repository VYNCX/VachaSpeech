import os
from contextlib import contextmanager

import torch
import torch.nn as nn

def freeze_modules(modules: list[nn.Module] | None):
    for module in modules:
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False

def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}

@contextmanager
def _suppress_stderr(enabled: bool):
    if not enabled:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull)


def _load_audio_internal(
    path: str, frame_offset: int | None = None, num_frames: int | None = None
) -> tuple[torch.Tensor, int]:

    import soundfile as sf

    suppress_warnings = _env_truthy("MIOCODEC_SUPPRESS_AUDIO_WARNINGS") or _env_truthy(
        "KANADE_TOKENIZER_SUPPRESS_AUDIO_WARNINGS"
    )
    with _suppress_stderr(suppress_warnings):
        with sf.SoundFile(path) as f:
            if frame_offset is not None:
                f.seek(frame_offset)
            frames = f.read(frames=num_frames or -1, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(frames.T)
            sample_rate = f.samplerate
    return waveform, sample_rate


def load_audio(audio_path: str, sample_rate: int = 24000) -> torch.Tensor:
    import torchaudio

    """Load and preprocess audio file."""
    waveform, sr = _load_audio_internal(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Normalize waveform
    max_val = torch.max(torch.abs(waveform)) + 1e-8
    waveform = waveform / max_val  # Normalize to [-1, 1]

    return waveform.squeeze(0)  # Remove channel dimension


def vocode(vocoder, mel_spectrogram: torch.Tensor) -> torch.Tensor:
    """Convert mel spectrogram to waveform using the selected vocoder.
    Args:
        vocoder (nn.Module): Vocoder with a decode() method.
        mel_spectrogram (torch.Tensor): Input mel spectrogram tensor (..., n_mels, frame).
    Returns:
        torch.Tensor: Generated audio waveform tensor (..., samples).
    """
    mel_spectrogram = mel_spectrogram.to(torch.float32)  # Ensure mel spectrogram is in float32
    with torch.inference_mode():
        generated_waveform = vocoder.decode(mel_spectrogram)
    return generated_waveform
