# train_model.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # Support Vector Machine - СОВЕРШЕННО другой алгоритм!
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import time
import logging
import joblib
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_processed_data():
    """Загрузка обработанных данных"""
    try:
        logger.info("Загрузка обработанных данных")
        X_train = pd.read_csv('data/processed/X_train.csv')
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_train = pd.read_csv('data/processed/y_train.csv')
        y_test = pd.read_csv('data/processed/y_test.csv')

        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        logger.info(f"Данные загружены: X_train {X_train.shape}, X_test {X_test.shape}")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        raise


def train_random_forest(X_train, X_test, y_train, y_test):
    """Обучение модели Random Forest (ансамблевый метод на деревьях)"""

    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1
    }

    start_time = time.time()

    with mlflow.start_run(run_name="Random Forest"):
        logger.info("Обучение Random Forest...")

        # Логируем параметры
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("algorithm", "Ensemble (Bagging)")

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Предсказания и метрики
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Кросс-валидация
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

        training_time = time.time() - start_time

        # Логируем метрики
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std(),
            "training_time_seconds": training_time
        })

        # Логируем артефакты
        mlflow.log_artifact("data/processed/X_train.csv", "data")
        mlflow.log_artifact("data/processed/X_test.csv", "data")

        # Сохраняем модель
        model_path = "models/random_forest_model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, "models")

        # Создаем и сохраняем confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Random Forest')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix_rf.png')
        mlflow.log_artifact('confusion_matrix_rf.png')
        plt.close()

        logger.info(f"Random Forest обучен за {training_time:.2f} сек. Accuracy: {accuracy:.4f}")

        return model, accuracy, training_time


def train_svm(X_train, X_test, y_train, y_test):
    """Обучение модели Support Vector Machine (SVM) - СОВЕРШЕННО другой алгоритм!"""

    params = {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'random_state': 42,
        'probability': True
    }

    start_time = time.time()

    with mlflow.start_run(run_name="Support Vector Machine"):
        logger.info("Обучение SVM...")

        # Логируем параметры
        mlflow.log_params(params)
        mlflow.log_param("model_type", "SVM")
        mlflow.log_param("algorithm", "Maximum Margin Classifier")

        model = SVC(**params)
        model.fit(X_train, y_train)

        # Предсказания и метрики
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Кросс-валидация
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')  # Меньше cv, т.к. SVM медленнее

        training_time = time.time() - start_time

        # Логируем метрики
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std(),
            "training_time_seconds": training_time
        })

        # Логируем данные
        mlflow.log_artifact("data/processed/X_train.csv", "data")
        mlflow.log_artifact("data/processed/X_test.csv", "data")

        # Сохраняем модель
        model_path = "models/svm_model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, "models")

        # Confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
        plt.title('Confusion Matrix - SVM')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix_svm.png')
        mlflow.log_artifact('confusion_matrix_svm.png')
        plt.close()

        logger.info(f"SVM обучен за {training_time:.2f} сек. Accuracy: {accuracy:.4f}")

        return model, accuracy, training_time


def compare_models(results):
    """Сравнение результатов двух моделей"""
    print("\n" + "=" * 50)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 50)

    for model_name, result in results.items():
        acc, time = result
        print(f"{model_name}: Accuracy = {acc:.4f}, Time = {time:.2f} сек")

    # Определяем лучшую модель
    best_model = max(results.items(), key=lambda x: x[1][0])
    print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model[0]} с accuracy {best_model[1][0]:.4f}")


def main():
    """Основная функция для обучения моделей"""

    # Настройка MLflow
    mlflow.set_experiment("Wine Quality Classification")

    # Загрузка данных
    X_train, X_test, y_train, y_test = load_processed_data()

    results = {}

    try:
        # Обучаем Random Forest
        rf_model, rf_accuracy, rf_time = train_random_forest(X_train, X_test, y_train, y_test)
        results["Random Forest"] = (rf_accuracy, rf_time)

        # Обучаем SVM
        svm_model, svm_accuracy, svm_time = train_svm(X_train, X_test, y_train, y_test)
        results["SVM"] = (svm_accuracy, svm_time)

        # Сравниваем результаты
        compare_models(results)

        logger.info("Обучение всех моделей завершено!")

    except Exception as e:
        logger.error(f"Ошибка при обучении моделей: {e}")
        raise


if __name__ == "__main__":
    main()