import torchaudio

print("Downloading LibriSpeech dev-other into data directory...")
librispeech_dev_other = torchaudio.datasets.LIBRISPEECH("./data", "dev-other", download=True)

print("Downloading LibriLight 10h into data directory...")
librilight_1h = torchaudio.datasets.LibriLightLimited("./data", "1h", download=True)
