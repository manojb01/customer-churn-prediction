# Customer Churn Prediction System

A production-ready machine learning pipeline for predicting customer churn in e-commerce using Apache Airflow, MLflow, and Streamlit.

## Overview

This project implements an end-to-end ML pipeline that:
- Processes and validates customer data from e-commerce platforms
- Trains and optimizes multiple machine learning models
- Tracks experiments and model versions using MLflow
- Provides a user-friendly Streamlit interface for predictions
- Automates the entire workflow using Apache Airflow

## Features

- **Automated ML Pipeline**: Scheduled data processing, training, and deployment using Airflow
- **Model Tracking**: Complete experiment tracking and model versioning with MLflow
- **Data Validation**: Comprehensive data quality checks and validation rules
- **Interactive UI**: Streamlit application for making predictions and monitoring models
- **Multiple Algorithms**: Support for Random Forest, XGBoost, and Logistic Regression
- **Hyperparameter Optimization**: Bayesian optimization for model tuning
- **Model Monitoring**: Data drift detection and performance monitoring

## Tech Stack

- **Orchestration**: Apache Airflow 3.0 (Astronomer)
- **ML Tracking**: MLflow 3.6.0
- **ML Libraries**: scikit-learn, XGBoost, imbalanced-learn
- **UI**: Streamlit
- **Database**: PostgreSQL
- **Containerization**: Docker & Docker Compose

## Project Structure

```
.
├── dags/                    # Airflow DAG definitions
├── models/                  # ML model code and artifacts
│   ├── ml_pipeline.py      # Core ML pipeline implementation
│   ├── production/         # Production model artifacts
│   └── staging/            # Staging model artifacts
├── data/                    # Data storage
│   ├── raw/                # Raw input data
│   └── processed/          # Processed datasets
├── data_validation/         # Data validation scripts
├── utils/                   # Utility functions
├── configs/                 # Configuration files
│   └── config.yaml         # Main configuration
├── app/                     # Application code
│   └── ui/                 # Streamlit UI
├── tests/                   # Test suite
├── scripts/                 # Utility scripts
├── mixed_customers.csv      # Test data for bulk predictions
├── docker-compose.override.yml  # Docker services configuration
└── requirements.txt         # Python dependencies
```

## Getting Started

### Prerequisites

- Docker Desktop installed and running
- Astronomer CLI (Astro CLI) installed
- At least 4GB of available RAM

### Installation

1. Clone the repository:
```bash
git clone https://github.com/manojb01/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Copy the environment template and configure:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start the services:
```bash
astro dev start
```

This will start:
- Airflow webserver at http://localhost:8080
- MLflow UI at http://localhost:5001
- Streamlit app at http://localhost:8501
- PostgreSQL database for Airflow metadata
- PostgreSQL database for MLflow tracking

### Configuration

Edit `.env` file to customize:
- Database credentials
- MLflow settings
- Logging levels
- Email/Slack alerts

See `.env.example` for all available options.

### Usage

1. **Access Airflow UI** (http://localhost:8080):
   - Default credentials: admin/admin (Astronomer default)
   - Enable and trigger the `customer_churn_prediction` DAG

2. **Monitor Training** (http://localhost:5001):
   - View MLflow experiments
   - Compare model performance
   - Manage model versions

3. **Make Predictions** (http://localhost:8501):
   - Upload customer data or use the interactive form
   - For bulk predictions, use `mixed_customers.csv` (test data provided in root directory)
   - Get churn predictions with probability scores
   - View model performance metrics

## Model Training

The pipeline trains multiple models:
- **Random Forest**: Ensemble method with calibrated probabilities
- **Logistic Regression**: Linear baseline model
- **XGBoost**: Gradient boosting (optional)

Models are evaluated on:
- ROC-AUC score (primary metric)
- Precision, Recall, F1-score
- Classification reports and confusion matrices

## Data Requirements

Input data should include these features:
- Customer demographics (Gender, Marital Status)
- Account information (Tenure, City Tier)
- Behavioral data (Hours on App, Login Device)
- Transaction data (Order Count, Cashback Amount)
- Satisfaction metrics (Satisfaction Score, Complaints)

See `configs/config.yaml` for the complete list of expected columns.

## Development

### Running Tests
```bash
pytest tests/
```

### Adding New Features
1. Update the DAG in `dags/`
2. Add model logic in `models/ml_pipeline.py`
3. Update configuration in `configs/config.yaml`
4. Test locally with `astro dev start`

### Code Quality
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Write tests for new functionality
- Update documentation

## Deployment

### Local Development
```bash
astro dev start    # Start services
astro dev stop     # Stop services
astro dev restart  # Restart services
```

### Production Deployment
For deploying to Astronomer Cloud:
```bash
astro deploy
```

See [Astronomer documentation](https://www.astronomer.io/docs/astro/deploy-code/) for detailed deployment instructions.

## Monitoring

The system includes:
- **Data Drift Detection**: Monitors input data distribution changes
- **Model Performance Tracking**: Alerts on metric degradation
- **Experiment Logging**: All runs logged to MLflow
- **Airflow Monitoring**: Task success/failure tracking

Configure alerts in `.env`:
- Email notifications (SMTP)
- Slack webhooks

## Troubleshooting

### Ports Already in Use
If ports 8080, 5001, or 8501 are occupied:
```bash
# Stop existing containers
docker ps
docker stop <container-id>
```

### Database Connection Issues
```bash
# Check service health
docker-compose ps
# View logs
docker-compose logs mlflow-db
```

### Model Training Failures
- Check Airflow task logs in the UI
- Verify data format matches `configs/config.yaml`
- Ensure sufficient memory (4GB+ recommended)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Open an issue on GitHub
- Review Astronomer guides at https://www.astronomer.io/docs/

## Acknowledgments

Built with Astronomer's Astro CLI and the open-source data community.
