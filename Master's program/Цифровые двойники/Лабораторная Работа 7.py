import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from enum import Enum, auto

warnings.filterwarnings('ignore')

class DataType(Enum):
    NUMERIC = auto()
    CATEGORICAL = auto()

@dataclass
class DataConfiguration:
    file_path: str = 'adult.data'
    delimiter: str = ','
    encoding: str = 'latin1'
    column_names: List[str] = field(
        default_factory=lambda: [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary'
        ]
    )
    target_column: str = 'salary'
    target_mapping: Dict[str, int] = field(
        default_factory=lambda: {' <=50K': 0, ' >50K': 1}
    )
    variant_range: Tuple[int, int] = field(
        default_factory=lambda: (VARIANT - 1, VARIANT)
    )

class DataLoader:
    def __init__(self, config: DataConfiguration):
        self.config = config
        
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(
            self.config.file_path,
            sep=self.config.delimiter,
            encoding=self.config.encoding,
            names=self.config.column_names
        )

class DataSplitter:
    @staticmethod
    def split_data(
        X: pd.DataFrame, 
        y: np.ndarray, 
        test_size: float = 0.3,
        random_state: int = SEED
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )

class DataAnalyzer(ABC):
    def __init__(self, config: DataConfiguration):
        self.config = config
        self.data_loader = DataLoader(config)
        
    @abstractmethod
    def analyze(self) -> None:
        pass

class NumericDataAnalyzer(DataAnalyzer):
    def analyze(self) -> None:
        data = self.data_loader.load_data()
        subset = data.iloc[
            self.config.variant_range[0] * 1000 : self.config.variant_range[1] * 1000
        ]
        X = subset.select_dtypes(exclude=['object']).copy()
        y = subset[self.config.target_column].map(self.config.target_mapping).values
        
        print("Первые 5 строк данных:")
        print(X.head())
        print("\nОписательная статистика:")
        print(X.describe())
        
        print("\nВизуализация взаимосвязей признаков:")
        DataVisualizer.plot_feature_relationships(X, y)
        
        X_train, X_test, y_train, y_test = DataSplitter.split_data(X, y)
        
        model_evaluator = ModelEvaluator()
        results = model_evaluator.evaluate_models(
            X_train, X_test, y_train, y_test,
            model_type=DecisionTreeClassifier,
            param_name='max_depth',
            param_range=range(1, 21)
        )
        
        print(f'\nОптимальные значения max_depth: {results.optimal_params}, точность: {results.best_score}')
        
        best_model = model_evaluator.train_best_model(
            X_train, y_train,
            model_type=DecisionTreeClassifier,
            param_name='max_depth',
            param_value=results.optimal_params[0])
        
        FeatureImportanceAnalyzer.plot_feature_importances(best_model, X.columns)

class CategoricalDataAnalyzer(DataAnalyzer):
    def __init__(self, config: DataConfiguration):
        super().__init__(config)
        self.converters: Dict[str, pd.Index] = {}
        
    def analyze(self) -> None:
        data = self.data_loader.load_data()
        subset = data.iloc[
            self.config.variant_range[0] * 1000 : self.config.variant_range[1] * 1000
        ]
        X = subset.select_dtypes(exclude=['object']).copy()
        y = subset[self.config.target_column].map(self.config.target_mapping).values
        
        categorical_cols = ['workclass', 'marital-status', 'occupation', 'native-country']
        for col in categorical_cols:
            X[col], self.converters[col] = pd.factorize(subset[col])
            
        X_train, X_test, y_train, y_test = DataSplitter.split_data(
            X, y, random_state=SEED**2)
        
        model_evaluator = ModelEvaluator()
        results = model_evaluator.evaluate_models(
            X_train, X_test, y_train, y_test,
            model_type=DecisionTreeClassifier,
            param_name='max_depth',
            param_range=range(1, 21))
        
        print(f'\nОптимальные значения max_depth (с категориальными признаками): {results.optimal_params}, точность: {results.best_score}')
        
        best_model = model_evaluator.train_best_model(
            X_train, y_train,
            model_type=DecisionTreeClassifier,
            param_name='max_depth',
            param_value=results.optimal_params[0])
        
        FeatureImportanceAnalyzer.plot_feature_importances(best_model, X.columns)

@dataclass
class EvaluationResults:
    optimal_params: List[int]
    best_score: float

class ModelEvaluator:
    def evaluate_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model_type: type,
        param_name: str,
        param_range: range,
        random_state: int = SEED
    ) -> EvaluationResults:
        cv_scores = []
        for param in param_range:
            model = model_type(**{param_name: param, 'random_state': random_state})
            model.fit(X_train, y_train)
            cv_scores.append(model.score(X_test, y_test))
            
        MSE = [1 - x for x in cv_scores]
        min_mse = min(MSE)
        optimal_params = [p for p, mse in zip(param_range, MSE) if mse <= min_mse]
        
        return EvaluationResults(optimal_params, 1 - min_mse)
        
    def train_best_model(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        model_type: type,
        param_name: str,
        param_value: int,
        random_state: int = SEED
    ) -> DecisionTreeClassifier:
        model = model_type(**{param_name: param_value, 'random_state': random_state})
        model.fit(X_train, y_train)
        return model

class DataVisualizer:
    @staticmethod
    def plot_feature_relationships(X: pd.DataFrame, y: np.ndarray) -> None:
        less = [row for row, v in zip(X.values, y) if v == 0]
        more = [row for row, v in zip(X.values, y) if v == 1]

        n_features = len(X.columns)
        h, w = int(np.sqrt(n_features)), int(np.ceil(n_features / int(np.sqrt(n_features))))
        
        fig, axes = plt.subplots(h, w, figsize=(15, 15))
        axes = axes.ravel()
        
        for i in range(n_features):
            for j in range(n_features):
                idx = i * n_features + j
                if idx >= len(axes):
                    continue
                    
                axes[idx].plot(
                    [row[i] for row in more],
                    [row[j] for row in more],
                    'go', label='>50K', markersize=4
                )
                axes[idx].plot(
                    [row[i] for row in less],
                    [row[j] for row in less],
                    'r^', label='<=50K', markersize=4
                )
                axes[idx].legend()
                axes[idx].set_xlabel(X.columns[i])
                axes[idx].set_ylabel(X.columns[j])
        
        plt.tight_layout()
        plt.show()

class FeatureImportanceAnalyzer:
    @staticmethod
    def plot_feature_importances(model: DecisionTreeClassifier, feature_names: pd.Index) -> None:
        importances = model.feature_importances_
        print("\nВажность признаков:", importances)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances)
        plt.xlabel('Признаки')
        plt.ylabel('Важность')
        plt.title('Важность признаков')
        plt.xticks(range(len(importances)), feature_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

class AnalysisPipeline:
    def __init__(self, config: DataConfiguration):
        self.config = config
        
    def run_analysis(self) -> None:
        print("=== Анализ числовых признаков ===")
        numeric_analyzer = NumericDataAnalyzer(self.config)
        numeric_analyzer.analyze()
        
        print("\n=== Анализ с категориальными признаками ===")
        categorical_analyzer = CategoricalDataAnalyzer(self.config)
        categorical_analyzer.analyze()

SEED = 42
VARIANT = 20

def main():
    config = DataConfiguration()
    pipeline = AnalysisPipeline(config)
    pipeline.run_analysis()

if __name__ == "__main__":
    main()