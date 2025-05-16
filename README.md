# RecBench

> Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation

## Installation

```bash
gh repo clone Jyonn/RecBench
cd RecBench
pip install -r requirements.txt
```

## 📊 Supported Datasets

RecBench supports 15 datasets across domains like news, books, movies, music, fashion, and e-commerce:

- 📰 MIND: Large-scale Microsoft news data for CTR prediction.
- 📰 PENS: Personalized news recommendation dataset.
- 📚 Goodreads: Book reviews and metadata.
- 📚 Amazon Books: Subset of Amazon product reviews.
- 🎥 MovieLens: Classic movie rating dataset.
- 📺 MicroLens: MovieLens dataset with user-item interactions.
- 📺 Netflix Prize: Large-scale movie rating competition dataset.
- 🎵 Amazon CDs: Music CD reviews and metadata.
- 🎵 Last.fm: Music playback logs and tagging data.
- 👗 H&M: Apparel and fashion product data.
- 👗 POG: Fashion product reviews and metadata.
- 📱 Amazon Electronics: Electronics product reviews and metadata.
- 🎮 Steam: Video game reviews and metadata.
- 🏨 HotelRec: Hotel recommendation dataset.
- ️️🍽️ Yelp: Restaurant reviews and metadata.

You can download our preprocessed data from [Kaggle (Recommended)](https://www.kaggle.com/datasets/qijiong/recbench/), [Google Drive](https://drive.google.com/drive/folders/1SocYpOWOxIWm7eB-CpFmyUR3Pb_-Pt9m?usp=sharing), and [Github Release](https://github.com/Jyonn/RecBench/releases/tag/dataset).

## Usage

### Example 1: Zero-shot, Pair-wise

```shell
python worker.py --model llama1 --data mind
```

### Example 2: Fine-tune, Pair-wise

```shell
python tuner.py --model llama1 --train mind --valid mind
```

### Example 3: Fine-tune, List-wise, Unique-ID-based

```shell
python seq_processor.py --data mind  # preprocess SeqRec data
python id_coder.py --data mind --seq true  # use unique identifier to represent items
python seq_tuner.py --model llama1seq --data mind --code_path ./code/mind.id.seq.code
```

### Example 4: Fine-tune, List-wise, Semantic-ID-based

```shell
python embedder.py --data mind --model llama1  # extract item embeddings 
python code_generator.py --data mind --model llama1  # use RQ-VAE for discrete tokenization
python seq_tuner.py --model llama1seq --data mind --code_path ./code/mind.llama1.seq.code
```

More documentations will be available soon.

## Updates

- **2025-03-07**: Our first benchmark paper is posted on arXiv: [Benchmarking LLMs in Recommendation Tasks: A Comparative Evaluation with Conventional Recommenders](https://arxiv.org/abs/2503.05493).
- **2024-12-15**: RecBench v1 library is released.
- **2024-06-04**: RecBench project is initiated.

## Citations

If you find RecBench useful in your research, please consider citing our project:

```
@article{liu2025benchmarking,
  title={Benchmarking LLMs in Recommendation Tasks: A Comparative Evaluation with Conventional Recommenders},
  author={Liu, Qijiong and Zhu, Jieming and Fan, Lu and Wang, Kun and Hu, Hengchang and Guo, Wei and Liu, Yong and Wu, Xiao-Ming},
  journal={arXiv preprint arXiv:2503.05493},
  year={2025}
}
```
