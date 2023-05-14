from lhotse.recipes import download_librispeech, prepare_librispeech

download_dir = "data/mini-librispeech"

# corpus_dir = download_librispeech(
#     target_dir=download_dir,
#     dataset_parts="mini_librispeech"
# )

prepare_librispeech(
    corpus_dir=f"{download_dir}/LibriSpeech",
    dataset_parts="mini_librispeech",
    output_dir=f"{download_dir}/manifests"
)
