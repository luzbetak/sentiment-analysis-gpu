Sentiment Analysis GPU
======================

A high-performance sentiment analysis project leveraging GPU acceleration for faster processing. This repository features a machine learning pipeline that processes and classifies text sentiment, optimized for GPU use to handle large datasets efficiently.

## Features

- **GPU-Accelerated Processing**: Utilizes CUDA-compatible GPUs for significantly faster analysis and training times.
- **Flexible Model Integration**: Supports various pre-trained models and custom configurations.
- **Scalable Architecture**: Capable of handling large-scale datasets for research or production use.
- **Comprehensive Sentiment Analysis**: Classifies text as positive, negative, or neutral with high accuracy.

## Requirements

- Python 3.x
- CUDA-compatible GPU
- PyTorch or TensorFlow with GPU support
- Required Python libraries listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/luzbetak/sentiment-analysis-gpu.git
   cd sentiment-analysis-gpu
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the sentiment analysis script using the following command:
```bash
python main.py --input data/sample_input.txt --use-gpu
```

For additional options, refer to the `--help` flag:
```bash
python main.py --help
```

## Project Structure

- **main.py**: Main script for running sentiment analysis.
- **models/**: Directory containing pre-trained and custom models.
- **data/**: Sample data files for testing and validation.
- **utils/**: Helper functions for preprocessing and evaluation.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request for any new features or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).
