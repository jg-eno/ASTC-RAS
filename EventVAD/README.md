# EventVAD: Training-Free Event-Aware Video Anomaly Detection (Accepted by ACM Multimedia 2025)

<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h5>

## News
- **[2025.07.09]** 🔥 We initiated this project and made both the code and data publicly available as open source.
- **[2025.07.06]** Our paper was accepted by ACM Multimedia 2025.

## Datasets

This project uses the following datasets for evaluation:

- **UCF-Crime**: A large-scale dataset for real-world surveillance video anomaly detection, containing 1900 long untrimmed videos with 13 real-world anomalies.  
  [Download link](https://www.crcv.ucf.edu/projects/real-world/)

- **XD-Violence**: A large-scale audio-visual dataset for violence detection in videos, comprising 4754 untrimmed videos covering 6 categories of anomalies.  
  [Download link](https://roc-ng.github.io/XD-Violence/)

Place the datasets in the directories specified by the code (e.g., `src/event_seg/videos` for input videos).

## Installation

This project requires Python 3.10. It is recommended to use a virtual environment to manage dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/YihuaJerry/EventVAD.git
   ```

2. Install the packages required for event splitting:
   ```bash
   conda create -n event_seg python=3.10
   conda activate event_seg
   cd src/event_seg
   pip install -r requirements.txt
   ```

3. Install the packages required for video scoring:
   ```bash
   conda create -n score python=3.10
   conda activate score
   cd src/score
   pip install -r requirements.txt
   ```

**Note**: Make sure to activate the virtual environment before installing dependencies. The 'evaluate.py' script relies on the library in 'score'.

### Local inference (uv, quantized)

The commands below create a Python 3.11 scoring environment. The official
VideoLLaMA2 checkpoint remains frozen; 4-bit NF4 quantization is applied while
loading it. This is a runtime quantization mode, not a separately
published quantized checkpoint. Keep model/data/output files out of git (they
are ignored by `.gitignore`).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu118 \
  'torch==2.2.0' 'torchvision==0.17.0'
uv pip install --python .venv/bin/python -r requirements-inference.txt
git clone https://github.com/DAMO-NLP-SG/VideoLLaMA2.git /tmp/VideoLLaMA2
git -C /tmp/VideoLLaMA2 checkout c0bb03abf6b8a6b9a8dccac006fb4db5d4d9e414
python scripts/patch_videollama2.py /tmp/VideoLLaMA2
uv pip install --python .venv/bin/python --no-deps -e /tmp/VideoLLaMA2
.venv/bin/huggingface-cli download DAMO-NLP-SG/VideoLLaMA2.1-7B-16F \
  --local-dir models/VideoLLaMA2.1-7B-16F
```

For a minimum scoring run, put paths and frame bounds in a whitespace-separated
manifest (`video_path start_frame end_frame`, one segment per line), then run:

```bash
.venv/bin/python src/score/event_score.py \
  --input_csv path/to/segment_manifest.txt \
  --output_csv outputs/scores.txt
```

This venv covers VideoLLaMA2 scoring. Event segmentation has separate LAVIS
constraints in the upstream project, so run that stage in its own environment
instead of mixing its older Transformers/timm pins into this scoring venv.
The default loader uses 4-bit NF4 and keeps the frozen model on CUDA. Use
`--quantization none` only when substantially more GPU memory is free. The
runner uses chunked SigLIP frame encoding and PyTorch SDPA to reduce peak
activation memory. `--model` accepts a local checkpoint directory or Hugging
Face model ID. Ensure enough GPU memory is free before starting; the weights
alone occupied about 7.2 GiB on the shared MIG slice during this setup.
The runner writes a `.details.csv` beside the score manifest with each raw
response and per-clip failure reason.

The repository's bundled `src/event_seg/videos/Robbery137_x264.mp4` is a smoke
input, not an evaluation dataset. Full UCF-Crime and XD-Violence downloads are
not bundled. Obtain them through the dataset authors' official access pages,
then point `src/event_seg/main.py --input` at the extracted videos. Event
segmentation also needs LAVIS/CLIP and the Princeton RAFT source/checkpoint; see
the upstream dependency constraints in `src/event_seg/requirements.txt` before
running that stage. Scoring the bundled sample directly requires a one-line
manifest containing its path and frame interval.

## Usage

To run the entire pipeline, follow these steps:

1. **Event Segmentation**:
   - Navigate to the event segmentation directory:
     ```bash
     cd src/event_seg
     ```
   - Run the main script:
     ```bash
     conda activate event_seg
     python main.py
     ```

2. **Scoring**:
   - Navigate to the scoring directory:
     ```bash
     cd src/score
     ```
   - Run the scoring script:
     ```bash
     conda activate score
     python event_score.py
     ```

3. **Evaluation**:
   - Run the evaluation script:
     ```bash
     cd src
     conda activate score
     python evaluate.py
     ```

**Note**: Ensure that the UCF-Crime and XD-Violence datasets are prepared and placed in the appropriate directories (e.g., `src/event_seg/videos` for input videos).

## Main Results
### UCF-Crime
<p><img src="./assets/performance(1).png" width="800" "/></p>

### XD-Violence
<p><img src="./assets/performance(2).png" width="800" "/></p>

## 📑 Citation

If you use this code or the EventVAD method in your research, please cite the following paper:

```bibtex
@article{shao2025eventvad,
  title={Eventvad: Training-free event-aware video anomaly detection},
  author={Shao, Yihua and He, Haojin and Li, Sijie and Chen, Siyu and Long, Xinwei and Zeng, Fanhu and Fan, Yuxuan and Zhang, Muyang and Yan, Ziyang and Ma, Ao and others},
  journal={arXiv preprint arXiv:2504.13092},
  year={2025}
}
```
