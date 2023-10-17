from dataclasses import dataclass
import soundfile
import random
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
import numpy as np
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
from pathlib import Path
from speechbrain.pretrained import EncoderClassifier

def get_yaapt_f0(audio: np.ndarray, rate=16000, interp=False) -> np.ndarray:
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate)
        pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25,
                                        'tda_frame_length': 25.0})
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]]

    f0 = np.vstack(f0s)
    return f0

@dataclass
class AudioSamplerConfig:
    code_sample_freq: int
    segment_size: int
    sampling_rate: int
    fmin: float
    fmax: float
    n_fft: int
    n_mels: int
    hop_size: int
    win_size: int

def dynamic_range_compression(x: torch.Tensor, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

class AudioSampler:
    def __init__(self, args: AudioSamplerConfig, embedder: EncoderClassifier):
        self.args = args
        self.embedder = embedder

        # A chirp is the length of a single quantized audio code.
        # Audio codes are sampled at 50Hz, so a chirp is 20ms long.
        # So we need to 20ms worth of audio to get a single code.
        self.chirp_len = self.args.sampling_rate // self.args.code_sample_freq
        self.chirps_per_segment = self.args.segment_size // self.chirp_len

        self.mel = T.MelSpectrogram(
            sample_rate=self.args.sampling_rate,
            n_fft=self.args.n_fft,
            n_mels=self.args.n_mels,
            f_min=self.args.fmin,
            f_max=self.args.sampling_rate // 2,
            win_length=self.args.win_size,
            hop_length=self.args.hop_size,
            window_fn=torch.hann_window,
            pad_mode="reflect",
            normalized=False
        )

    def process(self, audio: torch.Tensor, codes: torch.Tensor, sampling_ratio: float = 0.8):
        # Compute f0 over the whole audio clip.
        f0 = torch.tensor(get_yaapt_f0(audio[None].numpy(), self.args.sampling_rate, interp=False))[0, 0]
        f0_mean, f0_std = f0[f0 > 0.].mean(), f0[f0 > 0.].std()

        # Normalize f0 (where defined):
        f0_norm = (f0 - f0_mean) / f0_std
        f0_norm[f0 == 0.] = 0.

        n_f0s_per_chirp = round(self.chirp_len / (audio.shape[0] / f0.shape[0]))

        # Compute speaker embedding over the whole clip.
        speaker_embedding = self.embedder.encode_batch(audio[None], normalize=True)[0, 0]

        n_chirps = audio.shape[-1] // self.chirp_len
        chirp_idxs = torch.multinomial(
            torch.ones(n_chirps - self.chirps_per_segment),
            num_samples=int(sampling_ratio * n_chirps),
            replacement=False
        )
        f0_seg_range = torch.arange(n_f0s_per_chirp * self.chirps_per_segment)
        f0_idxs = f0_seg_range[None, :] + chirp_idxs[:, None] * n_f0s_per_chirp
        
        segs_f0 = f0_norm[f0_idxs]
        audio_seg_range = torch.arange(self.chirp_len * self.chirps_per_segment)
        audio_idxs = audio_seg_range[None, :] + chirp_idxs[:, None] * self.chirp_len
        segs_audio = audio[audio_idxs]

        segs_mel = self.mel(segs_audio.float())

        codes_seg_range = torch.arange(self.chirps_per_segment)
        codes_idxs = codes_seg_range[None, :] + chirp_idxs[:, None]
        segs_codes = codes[codes_idxs]
        return segs_f0, segs_audio, dynamic_range_compression(segs_mel), segs_codes, speaker_embedding[None].repeat(len(chirp_idxs), 1)
        #f0s = f0_norm[chirp_idxs * n_f0s_per_chirp: (chirp_idxs + self.chirps_per_segment) * n_f0s_per_chirp]
        #return f0s