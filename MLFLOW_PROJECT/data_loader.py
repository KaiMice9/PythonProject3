# data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import os
import joblib  # для сохранения скалера

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_process_data():
    """
    Полный пайплайн загрузки и обработки данных для Wine Quality
    """
    # 1. Загрузка данных
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    logger.info(f"Загрузка данных из {url}")

    df = pd.read_csv(url, delimiter=';')
    logger.info(f"Данные загружены. Форма: {df.shape}")

    # 2. Удаление дубликатов
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.info(f"Удаление {duplicates} дубликатов")
        df = df.drop_duplicates()
        logger.info(f"Данные после удаления дубликатов: {df.shape}")

    # 3. Проверка на пропущенные значения
    null_count = df.isnull().sum().sum()
    logger.info(f"Пропущенных значений: {null_count}")

    # 4. Feature Engineering - создаем новые признаки ДО разделения!
    logger.info("Создание новых признаков")
    df = create_new_features(df)

    # 5. Разделение на признаки и целевую переменную
    target_column = 'quality'
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 6. Разделение на train/test ДО нормализации (чтобы не было data leakage)
    logger.info("Разделение на train/test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Нормализация ОТДЕЛЬНО для train и test
    logger.info("Нормализация данных")
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # 8. Сохранение данных и артефактов
    save_artifacts(X_train_scaled, X_test_scaled, y_train, y_test, scaler)

    logger.info("Обработка данных завершена успешно!")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def create_new_features(df):
    """
    Создание новых признаков для улучшения качества модели
    """
    # Создаем новые признаки на основе доменного знания о вине
    df['acidity_ratio'] = df['fixed acidity'] / (df['volatile acidity'] + 1e-6)
    df['sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-6)
    df['alcohol_acidity_interaction'] = df['alcohol'] * df['fixed acidity']
    df['density_pH_interaction'] = df['density'] * df['pH']

    # Логарифмирование skewed features
    skewed_features = ['residual sugar', 'chlorides', 'sulphates']
    for feature in skewed_features:
        df[f'log_{feature}'] = np.log1p(df[feature])

    # Замена бесконечных значений
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())

    logger.info(f"Создано {len(skewed_features) + 4} новых признаков. Новая форма: {df.shape}")
    return df


def scale_data(X_train, X_test):
    """
    Нормализация данных отдельно для train и test
    """
    scaler = StandardScaler()

    # Обучаем scaler только на тренировочных данных!
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # применяем transform к test!

    # Преобразуем обратно в DataFrame с сохранением названий колонок
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, scaler


def save_artifacts(X_train, X_test, y_train, y_test, scaler):
    """
    Сохранение всех обработанных данных и артефактов
    """
    output_dir = 'data/processed'
    models_dir = 'models'

    try:
        # Создаем директории
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # Сохраняем данные
        X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
        X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
        y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
        y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

        # Сохраняем scaler для будущего использования
        joblib.dump(scaler, f'{models_dir}/scaler.joblib')

        # Сохраняем информацию о признаках
        feature_info = {
            'feature_names': X_train.columns.tolist(),
            'target_name': 'quality',
            'data_shape': {
                'X_train': X_train.shape,
                'X_test': X_test.shape,
                'y_train': y_train.shape,
                'y_test': y_test.shape
            }
        }
        joblib.dump(feature_info, f'{output_dir}/feature_info.joblib')

        logger.info(f"Данные сохранены в {output_dir}")
        logger.info(f"Scaler сохранен в {models_dir}/scaler.joblib")

    except Exception as e:
        logger.error(f"Ошибка при сохранении: {e}")
        raise


def get_data_info():
    """
    Получить информацию о сохраненных данных
    """
    try:
        feature_info = joblib.load('data/processed/feature_info.joblib')
        logger.info("Информация о данных:")
        logger.info(f"Признаки: {len(feature_info['feature_names'])}")
        logger.info(f"Размеры: {feature_info['data_shape']}")
        return feature_info
    except:
        logger.warning("Информация о данных не найдена")
        return None


if __name__ == "__main__":
    # Запуск полного пайплайна
    X_train, X_test, y_train, y_test, scaler = load_and_process_data()

    # Вывод информации
    print(f"\n✅ Обработка завершена!")
    print(f"📊 Train data: {X_train.shape}")
    print(f"📊 Test data: {X_test.shape}")
    print(f"🎯 Target distribution:")
    print(y_train.value_counts().sort_index())

    # Информация о новых признаках
    print(f"\n🆕 Новые признаки:")
    new_features = [col for col in X_train.columns if any(x in col for x in ['ratio', 'interaction', 'log_'])]
    for feature in new_features:
        print(f"  - {feature}")