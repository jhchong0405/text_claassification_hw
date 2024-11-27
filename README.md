This is my school project for the course "Text Mining" at the Peking University.

```markdown
# IMDB Sentiment Analysis

This project performs sentiment analysis on the IMDB dataset using both traditional machine learning and deep learning approaches. The results are logged and visualized to compare the performance of different models and configurations.

## Project Structure

- `analysis.py`: Main script to run experiments and generate plots.
- `ml_classifier.py`: Contains the `TraditionalMLClassifier` class for traditional machine learning models.
- `dl_classifier.py`: Contains the `DeepLearningClassifier` class for deep learning models using transformers.
- `plots/`: Directory where generated plots are saved.
- `logs/`: Directory where logs and results are saved.

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

Example:

```bash
python analysis.py --train-size 10000 --test-size 2000
```

## Results

- Results are logged in the `logs/` directory.
- Plots are saved in the `plots/` directory, including comparisons of model performance and parameter impacts.

## Troubleshooting

If you encounter issues related to random number generation or entropy sources, ensure your system's entropy source is available and properly configured. This is particularly relevant in containerized environments.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

```

### Notes:
- Replace `<repository-url>` and `<repository-directory>` with the actual URL and directory name of your repository.
- Ensure that you have a `requirements.txt` file listing all the necessary Python packages.
- Adjust the instructions based on your specific setup and any additional details relevant to your project.
