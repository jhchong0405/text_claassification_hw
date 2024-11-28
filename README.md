# IMDB Sentiment Analysis

This project performs sentiment analysis on the IMDB dataset using both traditional machine learning and deep learning approaches. The results are logged and visualized to compare the performance of different models and configurations.

## Project Structure

- `analysis.py`: Main script to run experiments and generate plots.
- `ml_classifier.py`: Contains the `TraditionalMLClassifier` class for traditional machine learning models.
- `dl_classifier.py`: Contains the `DeepLearningClassifier` class for deep learning models using transformers.
- `plots/`: Directory where generated plots are saved.
- `logs/`: Directory where logs and results are saved.
- `test.sh`: Shell script to automate running experiments with different dataset sizes.

## Setup

### Prerequisites

- Python 3.7 or higher
- Virtual environment (recommended)
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Requirements

Ensure you have the following packages installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `torch`
- `transformers`
- `tensorflow`
- `tqdm`

## Usage

### Running Experiments

To run the experiments and generate plots, execute the following command:

```bash
python analysis.py
```

### Options

- `--show-only`: Only display results without running new experiments.
- `--train-size`: Specify the size of the training dataset (default: 5000).
- `--test-size`: Specify the size of the testing dataset (default: 1000).
- `--save-checkpoints`: Save model checkpoints during training.

Example:

```bash
python analysis.py --train-size 10000 --test-size 2000
```

### Running Experiments with `test.sh`

You can automate running experiments with different dataset sizes using the `test.sh` script:

```bash
chmod +x test.sh
./test.sh
```

This script will run experiments for predefined train and test sizes.

## Results

- Results are logged in the `logs/` directory.
- Plots are saved in the `plots/` directory, including comparisons of model performance, parameter impacts, and detailed metrics.
- A summary of the best models and their metrics is saved in `plots/results_summary.txt`.

## Troubleshooting

If you encounter issues related to random number generation or entropy sources, ensure your system's entropy source is available and properly configured. This is particularly relevant in containerized environments.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

</rewritten_file>
