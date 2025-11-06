## Important Note: This module is primarily designed for simple datasets, such as the iris dataset or any of the UCI repository datasets, for analysis. For cases where you have more features and want to perform advanced feature engineering to achieve the best results, use the IDBNN module. The DBNN code remains the same, but the pre-processing steps vary slightly. 

```
Use python adaptive_dbnn.py --gui for setting up the configuration.

{
    "file_path": "breast_cancer_wisconsin.csv",
    "target_column": "diagnosis",
    "separator": ",",
    "has_header": true,
    "training_config": {
        "trials": 100,
        "cardinality_threshold": 0.9,
        "cardinality_tolerance": 4,
        "learning_rate": 0.1,
        "random_seed": 42,
        "epochs": 1000,
        "test_fraction": 0.2,
        "train": true,
        "train_only": false,
        "predict": true,
        "gen_samples": false
    },
    "likelihood_config": {
        "feature_group_size": 2,
        "max_combinations": 1000,
        "update_strategy": 3
    },
    "adaptive_learning": {
        "enable_adaptive": true,
        "initial_samples_per_class": 10,
        "margin": 0.15,
        "max_adaptive_rounds": 15
    },
    "statistics": {
        "enable_confusion_matrix": true,
        "enable_progress_plots": true,
        "color_progress": "green",
        "color_regression": "red",
        "save_plots": true
    }
}
```
