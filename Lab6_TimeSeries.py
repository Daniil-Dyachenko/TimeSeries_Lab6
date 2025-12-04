"""
Лабораторна робота №6: Аналіз та обробка часових рядів (Time Series)
ЕКСПОНЕНЦІАЛЬНЕ ЗГЛАДЖУВАННЯ ДЛЯ ОБРОБКИ ТА ПРОГНОЗУВАННЯ

Завдання ІІ рівня – максимально 10 балів.
Реалізувати групу вимог 2

Група вимог 2:
1.Обробка вхідних даних лр1,2, або самостійно обраних з використанням регресійних моделей:
2.Дослідження даних та виявлення властивостей;
3.Оптимальний вибір моделі обробки – типу алгоритму експоненціального згладжування;
4.Обробку та екстраполяцію вхідних даних;
5.Оцінювання КРІ моделі, результатів обробки та екстраполяції;
6.Візуалізація / побудова графіків вхідних даних та результатів обробки / екстраполяції.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os
import requests
import re
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 50

def parse_minfin_living_wage(url='https://index.minfin.com.ua/ua/labour/wagemin/',
                             use_backup=True, save_data=True):
    """Парсинг прожиткового мінімуму з сайту Minfin"""
    print("-" * 80)
    print("ПАРСИНГ ДАНИХ З САЙТУ MINFIN.COM.UA")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'uk-UA,uk;q=0.9',
    }

    try:
        print(f"Спроба парсингу: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Статус відповіді: {response.status_code}")

        if response.status_code == 200:
            df_list = pd.read_html(response.text)
            if df_list and len(df_list) > 0:
                df = df_list[0]
                print(f"Знайдено таблицю з {len(df)} записами")
                parsed_data = process_minfin_data(df)

                if save_data:
                    save_parsed_data(parsed_data)
                return parsed_data

        print("Не вдалося знайти таблицю на сторінці")

    except Exception as e:
        print(f"Помилка парсингу: {e}")

    if use_backup:
        print("\nВикористовуємо резервні реалістичні дані")
        backup_data = generate_realistic_minfin_data()
        if save_data:
            save_parsed_data(backup_data)
        return backup_data
    else:
        raise Exception("Парсинг не вдався і резервні дані вимкнено")


def process_minfin_data(df):
    """Обробка даних після парсингу"""
    df = df.dropna()
    df.columns = ['period', 'total', 'children_under_6', 'children_6_18',
                  'working_age', 'disabled']

    dates = []
    values_total = []

    for idx, row in df.iterrows():
        period_str = str(row['period'])
        total_val = row['total']
        match = re.search(r'з\s+(\d{2}\.\d{2}\.\d{4})', period_str)

        if match:
            date_str = match.group(1)
            try:
                date = pd.to_datetime(date_str, format='%d.%m.%Y')
                dates.append(date)
                values_total.append(float(total_val))
            except:
                continue

    if dates:
        df_processed = pd.DataFrame({
            'date': dates,
            'living_wage': values_total
        })
        df_processed = df_processed.sort_values('date').reset_index(drop=True)

        print(f"Дані оброблено: {len(df_processed)} записів")
        print(f"Період: {df_processed['date'].min().date()} – {df_processed['date'].max().date()}")
        print(f"Діапазон: {df_processed['living_wage'].min():.0f} – {df_processed['living_wage'].max():.0f} грн")

        return df_processed
    else:
        raise Exception("Не вдалося обробити дати з таблиці")


def generate_realistic_minfin_data():
    """Генерація реалістичних даних (резервний варіант)"""
    dates = pd.date_range(start='2000-01-01', end='2025-01-01', freq='QS')
    n = len(dates)

    base_value = 270
    years_passed = np.arange(n) / 4
    growth_rate = 0.08

    living_wage = base_value * np.exp(growth_rate * years_passed)
    noise = np.random.normal(0, 20, n)
    living_wage = living_wage + noise
    living_wage = np.round(living_wage).astype(int)

    df = pd.DataFrame({
        'date': dates,
        'living_wage': living_wage
    })

    print(f"Згенеровано {len(df)} записів реалістичних даних")
    print(f"Період: {dates[0].date()} – {dates[-1].date()}")
    print(f"Діапазон: {living_wage.min()} – {living_wage.max()} грн")

    return df


def save_parsed_data(df, directory='parsed_data'):
    """Збереження спарсених даних"""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, "parsed_minfin_data.csv")
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"Дані збережено: {filepath}")


def synthesize_data(n=100, trend_type='exponential', seasonal_amplitude=100, add_anomalies=False):
    """Синтез даних з різними типами трендів для тестування"""
    print("-" * 80)
    print("СИНТЕЗ ТЕСТОВИХ ДАНИХ")

    dates = pd.date_range(start='2020-01-01', periods=n, freq='M')
    x = np.arange(n)

    if trend_type == 'exponential':
        base = 2000
        growth_rate = 0.015
        trend = base * np.exp(growth_rate * x)

        seasonal = seasonal_amplitude * np.sin(2 * np.pi * x / 12)
        noise = np.random.normal(0, seasonal_amplitude * 0.1, n)
        values = trend + seasonal + noise

        print(f"Синтезовано {n} записів")
        print(f"Тип тренду: {trend_type}")
        print(f"Амплітуда сезонності: {seasonal_amplitude}")

    elif trend_type == 'linear':
        trend = 2000 + 30 * x
        seasonal = seasonal_amplitude * np.sin(2 * np.pi * x / 12)
        noise = np.random.normal(0, seasonal_amplitude * 0.1, n)
        values = trend + seasonal + noise

        if add_anomalies:
            n_anomalies = max(1, int(n * 0.08))
            Q_AV = 5
            anomaly_indices = np.random.choice(n, n_anomalies, replace=False)

            for idx in anomaly_indices:
                values[idx] += np.random.choice([-1, 1]) * Q_AV * (seasonal_amplitude * 0.1)

            print(f"Додано {n_anomalies} аномальних вимірів ({n_anomalies / n * 100:.1f}%)")

        print(f"Синтезовано {n} записів")
        print(f"Тип тренду: {trend_type}")
        print(f"Амплітуда сезонності: {seasonal_amplitude}")

    elif trend_type == 'cubic':

        x_scaled = x / (n / 100) if n > 100 else x

        a0, a1, a2, a3 = 2000, 30, 0.5, 0.002
        trend = a0 + a1 * x_scaled + a2 * x_scaled ** 2 + a3 * x_scaled ** 3

        noise_std = np.std(trend) * 0.05
        noise = np.random.normal(0, noise_std, n)

        values = trend + noise

        n_anomalies = max(1, int(n * 0.10))
        Q_AV = 6
        anomaly_indices = np.random.choice(n, n_anomalies, replace=False)

        for idx in anomaly_indices:
            values[idx] += np.random.choice([-1, 1]) * Q_AV * noise_std

        print(f"Синтезовано {n} записів")
        print(f"Тип тренду: кубічний")
        print(f"Коефіцієнти: a0={a0}, a1={a1}, a2={a2}, a3={a3}")
        print(f"Нормальний шум: μ=0, σ={noise_std:.2f}")
        print(f"Додано {n_anomalies} аномальних вимірів ({n_anomalies / n * 100:.1f}%)")

    else:
        seasonal = 2000 + seasonal_amplitude * np.sin(2 * np.pi * x / 12)
        noise = np.random.normal(0, seasonal_amplitude * 0.1, n)
        values = seasonal + noise

        print(f"Синтезовано {n} записів")
        print(f"Тип тренду: seasonal only")
        print(f"Амплітуда сезонності: {seasonal_amplitude}")

    df = pd.DataFrame({
        'date': dates,
        'living_wage': values
    })

    print(f"Діапазон: {values.min():.0f} – {values.max():.0f}")
    print(f"Середнє: {values.mean():.2f}, СКВ: {values.std():.2f}")

    return df

def analyze_time_series_properties(data, column='living_wage'):
    """Дослідження властивостей часового ряду"""
    print("\n" + "-" * 80)
    print("ДОСЛІДЖЕННЯ ВЛАСТИВОСТЕЙ ЧАСОВОГО РЯДУ")

    series = data[column].values
    n = len(series)

    print("\nСТАТИСТИЧНІ ХАРАКТЕРИСТИКИ:")
    print(f"Обсяг вибірки: {n}")
    print(f"Середнє значення: {np.mean(series):.2f}")
    print(f"Медіана: {np.median(series):.2f}")
    print(f"СКВ: {np.std(series):.2f}")
    print(f"Мінімум: {np.min(series):.2f}")
    print(f"Максимум: {np.max(series):.2f}")
    print(f"Розмах: {np.max(series) - np.min(series):.2f}")

    print("\nАНАЛІЗ ТРЕНДУ:")
    x = np.arange(n)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)

    print(f"Коефіцієнт нахилу: {slope:.4f}")
    print(f"R²: {r_value ** 2:.4f}")
    print(f"p-value: {p_value:.4e}")

    if p_value < 0.05:
        if slope > 0:
            print("Зростаючий тренд")
        else:
            print("Спадний тренд ")
    else:
        print("Тренд відсутній або незначущий")

    print("\nАНАЛІЗ СЕЗОННОСТІ:")
    if n >= 24:
        lags_to_test = [3, 4, 6, 12]
        significant_lags = []

        for lag in lags_to_test:
            if lag < n:
                acf_value = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                threshold = 1.96 / np.sqrt(n)

                if abs(acf_value) > threshold:
                    significant_lags.append((lag, acf_value))
                    print(f"Lag {lag}: ACF = {acf_value:.3f} (значущий)")

    print("\nАНАЛІЗ СТАЦІОНАРНОСТІ:")

    diff_series = np.diff(series)

    var_original = np.var(series)
    var_diff = np.var(diff_series)

    print(f"Дисперсія оригінального ряду: {var_original:.2f}")
    print(f"Дисперсія після диференціювання: {var_diff:.2f}")

    if var_diff < var_original * 0.5:
        print("Ряд НЕСТАЦІОНАРНИЙ ")
    else:
        print("Ряд ближче до СТАЦІОНАРНОГО")

    properties = {
        'n': n,
        'mean': np.mean(series),
        'std': np.std(series),
        'has_trend': p_value < 0.05,
        'trend_type': 'increasing' if slope > 0 else 'decreasing' if p_value < 0.05 else 'none',
        'has_seasonality': len(significant_lags) > 0 if n >= 24 else False,
        'is_stationary': var_diff >= var_original * 0.5
    }

    return properties

def select_optimal_smoothing_model(data, column='living_wage', properties=None):
    """Вибір оптимального типу експоненціального згладжування"""
    print("\n" + "-" * 80)
    print("ОПТИМАЛЬНИЙ ВИБІР МОДЕЛІ ЕКСПОНЕНЦІАЛЬНОГО ЗГЛАДЖУВАННЯ")

    series = data[column].values
    n = len(series)

    split_point = int(n * 0.8)
    train = series[:split_point]
    test = series[split_point:]

    print(f"\nРозмір навчальної вибірки: {len(train)}")
    print(f"Розмір тестової вибірки: {len(test)}")

    models_to_test = []

    print("\nТестування Simple Exponential Smoothing (SES)...")
    try:
        model_ses = SimpleExpSmoothing(train).fit()
        forecast_ses = model_ses.forecast(len(test))
        mae_ses = mean_absolute_error(test, forecast_ses)
        rmse_ses = np.sqrt(mean_squared_error(test, forecast_ses))

        print(f"MAE: {mae_ses:.2f}")
        print(f"RMSE: {rmse_ses:.2f}")
        print(f"Alpha (α): {model_ses.params['smoothing_level']:.4f}")

        models_to_test.append({
            'name': 'Simple Exponential Smoothing',
            'short_name': 'SES',
            'model': model_ses,
            'mae': mae_ses,
            'rmse': rmse_ses,
            'params': {'alpha': model_ses.params['smoothing_level']}
        })
    except Exception as e:
        print(f"Помилка: {e}")

    print("\nТестування Holt's Linear Trend(Подвійне згладжування):")
    try:
        model_holt = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
        forecast_holt = model_holt.forecast(len(test))
        mae_holt = mean_absolute_error(test, forecast_holt)
        rmse_holt = np.sqrt(mean_squared_error(test, forecast_holt))

        print(f"MAE: {mae_holt:.2f}")
        print(f"RMSE: {rmse_holt:.2f}")
        print(f"Alpha (α): {model_holt.params['smoothing_level']:.4f}")
        print(f"Beta (β): {model_holt.params['smoothing_trend']:.4f}")

        models_to_test.append({
            'name': "Holt's Linear Trend",
            'short_name': 'Holt',
            'model': model_holt,
            'mae': mae_holt,
            'rmse': rmse_holt,
            'params': {
                'alpha': model_holt.params['smoothing_level'],
                'beta': model_holt.params['smoothing_trend']
            }
        })
    except Exception as e:
        print(f"Помилка: {e}")

    if n >= 24:
        print("\nТестування Holt-Winters(Потрійне згладжування):")

        try:
            model_hw_add = ExponentialSmoothing(
                train,
                trend='add',
                seasonal='add',
                seasonal_periods=min(12, len(train) // 3)
            ).fit()
            forecast_hw_add = model_hw_add.forecast(len(test))
            mae_hw_add = mean_absolute_error(test, forecast_hw_add)
            rmse_hw_add = np.sqrt(mean_squared_error(test, forecast_hw_add))

            print(f"Адитивна сезонність:")
            print(f"MAE: {mae_hw_add:.2f}")
            print(f"RMSE: {rmse_hw_add:.2f}")

            models_to_test.append({
                'name': 'Holt-Winters (Additive)',
                'short_name': 'HW_Add',
                'model': model_hw_add,
                'mae': mae_hw_add,
                'rmse': rmse_hw_add,
                'params': {
                    'alpha': model_hw_add.params['smoothing_level'],
                    'beta': model_hw_add.params['smoothing_trend'],
                    'gamma': model_hw_add.params['smoothing_seasonal']
                }
            })
        except Exception as e:
            print(f"Помилка: {e}")

        if np.all(train > 0):
            try:
                model_hw_mul = ExponentialSmoothing(
                    train,
                    trend='add',
                    seasonal='mul',
                    seasonal_periods=min(12, len(train) // 3)
                ).fit()
                forecast_hw_mul = model_hw_mul.forecast(len(test))
                mae_hw_mul = mean_absolute_error(test, forecast_hw_mul)
                rmse_hw_mul = np.sqrt(mean_squared_error(test, forecast_hw_mul))

                print(f"Мультиплікативна сезонність:")
                print(f"MAE: {mae_hw_mul:.2f}")
                print(f"RMSE: {rmse_hw_mul:.2f}")

                models_to_test.append({
                    'name': 'Holt-Winters (Multiplicative)',
                    'short_name': 'HW_Mul',
                    'model': model_hw_mul,
                    'mae': mae_hw_mul,
                    'rmse': rmse_hw_mul,
                    'params': {
                        'alpha': model_hw_mul.params['smoothing_level'],
                        'beta': model_hw_mul.params['smoothing_trend'],
                        'gamma': model_hw_mul.params['smoothing_seasonal']
                    }
                })
            except Exception as e:
                print(f" Помилка: {e}")
    else:
        print("\n3. Holt-Winters недоступний, потрібно більше спостережень")

    print("\n" + "-" * 80)
    print("ПОРІВНЯННЯ МОДЕЛЕЙ:")
    print("-" * 80)
    print(f"{'Модель':<35} {'MAE':>12} {'RMSE':>12}")
    print("-" * 80)

    for model_info in models_to_test:
        print(f"{model_info['name']:<35} {model_info['mae']:>12.2f} {model_info['rmse']:>12.2f}")

    best_model = min(models_to_test, key=lambda x: x['mae'])

    print("\n" + "-" * 80)
    print(f"ОПТИМАЛЬНА МОДЕЛЬ: {best_model['name']}")
    print(f"MAE: {best_model['mae']:.2f}")
    print(f"RMSE: {best_model['rmse']:.2f}")
    print(f"\nПараметри моделі:")
    for param, value in best_model['params'].items():
        print(f"  {param}: {value:.4f}")

    return best_model, models_to_test, train, test

def train_and_forecast(data, model_info, forecast_periods=10, column='living_wage'):
    """Навчання на повних даних та прогнозування"""
    print("\n" + "-" * 80)
    print("НАВЧАННЯ ПОВНОЇ МОДЕЛІ ТА ЕКСТРАПОЛЯЦІЯ")

    series = data[column].values

    print(f"\nНавчання моделі '{model_info['name']}' на {len(series)} спостереженнях...")

    model_type = model_info['short_name']

    try:
        if model_type == 'SES':
            final_model = SimpleExpSmoothing(series).fit()

        elif model_type == 'Holt':
            final_model = ExponentialSmoothing(series, trend='add', seasonal=None).fit()

        elif model_type == 'HW_Add':
            seasonal_periods = min(12, len(series) // 3)
            final_model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods
            ).fit()

        elif model_type == 'HW_Mul':
            seasonal_periods = min(12, len(series) // 3)
            final_model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='mul',
                seasonal_periods=seasonal_periods
            ).fit()

        fitted = final_model.fittedvalues

        forecast = final_model.forecast(forecast_periods)

        print(f"Модель успішно навчена")
        print(f"Прогноз на {forecast_periods} періодів")
        print(f"\nПрогнозні значення:")
        for i, val in enumerate(forecast, 1):
            print(f"Період +{i}: {val:.2f}")

        mae_train = mean_absolute_error(series, fitted)
        rmse_train = np.sqrt(mean_squared_error(series, fitted))

        mape_train = np.mean(np.abs((series - fitted) / series)) * 100

        print(f"\nМетрики на навчальних даних:")
        print(f"  MAE: {mae_train:.2f}")
        print(f"  RMSE: {rmse_train:.2f}")
        print(f"  MAPE: {mape_train:.2f}%")

        return {
            'model': final_model,
            'fitted': fitted,
            'forecast': forecast,
            'forecast_periods': forecast_periods,
            'metrics': {
                'mae': mae_train,
                'rmse': rmse_train,
                'mape': mape_train
            }
        }

    except Exception as e:
        print(f"Помилка навчання моделі: {e}")
        return None

def evaluate_model_kpi(data, trained_model_info, test_data=None, column='living_wage'):
    """Комплексна оцінка KPI моделі"""
    print("\n" + "-" * 80)
    print("ОЦІНЮВАННЯ KPI МОДЕЛІ")

    series = data[column].values
    fitted = trained_model_info['fitted']
    residuals = series - fitted

    print("\nМЕТРИКИ ТОЧНОСТІ:")
    print(f"MAE (Mean Absolute Error): {trained_model_info['metrics']['mae']:.2f}")
    print(f"RMSE (Root Mean Squared Error): {trained_model_info['metrics']['rmse']:.2f}")
    print(f"MAPE (Mean Absolute Percentage Error): {trained_model_info['metrics']['mape']:.2f}%")

    print("\nАНАЛІЗ ЗАЛИШКІВ:")
    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals)

    print(f"Середнє залишків: {mean_residuals:.4f}")
    print(f"СКВ залишків: {std_residuals:.2f}")

    if len(residuals) <= 5000:
        stat, p_value = stats.shapiro(residuals)
        print(f"Тест Shapiro-Wilk (нормальність): p-value = {p_value:.4f}")
        if p_value > 0.05:
            print("Залишки нормально розподілені")
        else:
            print("Залишки НЕ нормально розподілені")

    print("\nАВТОКОРЕЛЯЦІЯ ЗАЛИШКІВ:")
    if len(residuals) > 10:
        lag1_corr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        print(f"Lag-1 автокореляція: {lag1_corr:.4f}")

        threshold = 1.96 / np.sqrt(len(residuals))
        if abs(lag1_corr) < threshold:
            print(f"Автокореляція незначуща (порог: ±{threshold:.4f})")
        else:
            print(f"Виявлена значуща автокореляція (можливе покращення моделі)")

    print("\nКОЕФІЦІЄНТ ДЕТЕРМІНАЦІЇ:")
    ss_res = np.sum((series - fitted) ** 2)
    ss_tot = np.sum((series - np.mean(series)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"R² = {r2:.4f}")

    print("\nІНФОРМАЦІЙНІ КРИТЕРІЇ:")
    n = len(series)
    k = len(trained_model_info['model'].params)

    aic = n * np.log(ss_res / n) + 2 * k
    bic = n * np.log(ss_res / n) + k * np.log(n)

    print(f"AIC (Akaike Information Criterion): {aic:.2f}")
    print(f"BIC (Bayesian Information Criterion): {bic:.2f}")

    kpi = {
        'mae': trained_model_info['metrics']['mae'],
        'rmse': trained_model_info['metrics']['rmse'],
        'mape': trained_model_info['metrics']['mape'],
        'r2': r2,
        'aic': aic,
        'bic': bic,
        'mean_residuals': mean_residuals,
        'std_residuals': std_residuals
    }

    return kpi

def visualize_results(data, trained_model_info, properties, model_name,
                      column='living_wage', save_dir='Lab6_Graphs'):
    """Комплексна візуалізація всіх результатів"""
    print("\n" + "-" * 80)
    print("ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ")

    os.makedirs(save_dir, exist_ok=True)

    series = data[column].values
    dates = data['date'].values if 'date' in data.columns else np.arange(len(series))
    fitted = trained_model_info['fitted']
    forecast = trained_model_info['forecast']
    forecast_periods = trained_model_info['forecast_periods']

    if 'date' in data.columns:
        last_date = pd.to_datetime(data['date'].iloc[-1])
        freq = pd.infer_freq(data['date'])
        if freq is None:
            freq = 'M'
        forecast_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq=freq)[1:]
    else:
        forecast_dates = np.arange(len(series), len(series) + forecast_periods)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(dates, series, 'b-', linewidth=1.5, label='Вихідні дані')
    axes[0, 0].set_title('Вихідний часовий ряд', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Дата/Час')
    axes[0, 0].set_ylabel('Значення')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(series, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Розподіл значень', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Значення')
    axes[0, 1].set_ylabel('Частота')
    axes[0, 1].axvline(np.mean(series), color='red', linestyle='--',
                       linewidth=2, label=f'Середнє: {np.mean(series):.2f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if len(series) > 10:
        plot_acf(series, lags=min(40, len(series) // 2), ax=axes[1, 0], alpha=0.05)
        axes[1, 0].set_title('Автокореляційна функція (ACF)', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

    diff_series = np.diff(series)
    axes[1, 1].plot(dates[1:], diff_series, 'g-', linewidth=1, alpha=0.7)
    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1, 1].set_title('Перші різниці (диференціювання)', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Дата/Час')
    axes[1, 1].set_ylabel('Значення')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('АНАЛІЗ ВЛАСТИВОСТЕЙ ВИХІДНИХ ДАНИХ',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    filename = f'1_data_properties.png'
    fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Збережено: {save_dir}/{filename}")
    plt.close()

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(dates, series, 'b-', linewidth=1.5, alpha=0.6,
            label='Вихідні дані', marker='o', markersize=3)

    ax.plot(dates, fitted, 'g-', linewidth=2.5,
            label=f'Згладжені дані ({model_name})', alpha=0.9)

    ax.plot(forecast_dates, forecast, 'r-', linewidth=2.5,
            label='Прогноз (екстраполяція)', marker='s', markersize=5)

    forecast_std = np.std(series - fitted)
    lower_bound = forecast - 1.96 * forecast_std
    upper_bound = forecast + 1.96 * forecast_std

    ax.fill_between(forecast_dates, lower_bound, upper_bound,
                    color='red', alpha=0.2, label='95% довірчий інтервал')

    ax.axvline(dates[-1], color='orange', linestyle='--',
               linewidth=2, alpha=0.7, label='Межа даних')

    ax.set_title(f'ЕКСПОНЕНЦІАЛЬНЕ ЗГЛАДЖУВАННЯ: {model_name}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Дата/Час', fontsize=12)
    ax.set_ylabel('Значення', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = f'2_smoothing_and_forecast_{model_name.replace(" ", "_")}.png'
    fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Збережено: {save_dir}/{filename}")
    plt.close()

    residuals = series - fitted

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(dates, residuals, 'b-', linewidth=1, alpha=0.7)
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Залишки у часі', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Дата/Час')
    axes[0, 0].set_ylabel('Залишки')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(residuals, bins=30, color='steelblue',
                    alpha=0.7, edgecolor='black', density=True)

    mu, sigma = np.mean(residuals), np.std(residuals)
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 1].plot(x_norm, stats.norm.pdf(x_norm, mu, sigma),
                    'r-', linewidth=2, label='Нормальний розподіл')

    axes[0, 1].set_title('Розподіл залишків', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Залишки')
    axes[0, 1].set_ylabel('Щільність')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (перевірка нормальності)',
                         fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(fitted, residuals, alpha=0.5, s=20, color='steelblue')
    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Залишки vs Згладжені значення',
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Згладжені значення')
    axes[1, 1].set_ylabel('Залишки')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f'АНАЛІЗ ЗАЛИШКІВ - {model_name}',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    filename = f'3_residuals_analysis_{model_name.replace(" ", "_")}.png'
    fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Збережено: {save_dir}/{filename}")
    plt.close()

    if hasattr(trained_model_info['model'], 'level') and hasattr(trained_model_info['model'], 'trend'):
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        axes[0].plot(dates, trained_model_info['model'].level,
                     'b-', linewidth=2, label='Рівень (Level)')
        axes[0].set_title('Компонента рівня', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Значення')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(dates, trained_model_info['model'].trend,
                     'g-', linewidth=2, label='Тренд')
        axes[1].set_title('Компонента тренду', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Значення')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        if hasattr(trained_model_info['model'], 'season'):
            season_component = trained_model_info['model'].season
            axes[2].plot(dates[-len(season_component):], season_component,
                         'r-', linewidth=2, label='Сезонність')
            axes[2].set_title('Сезонна компонента', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Дата/Час')
            axes[2].set_ylabel('Значення')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'Сезонна компонента відсутня',
                         ha='center', va='center', fontsize=14)
            axes[2].set_title('Сезонна компонента', fontsize=12, fontweight='bold')

        fig.suptitle(f'ДЕКОМПОЗИЦІЯ МОДЕЛІ - {model_name}',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        filename = f'4_model_decomposition_{model_name.replace(" ", "_")}.png'
        fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Збережено: {save_dir}/{filename}")
        plt.close()

    print(f"\nУсі графіки збережено в папці: {save_dir}/")

def main():
    """Головна функція програми"""
    print("\n" + "-" * 80)
    print("           ЛАБОРАТОРНА РОБОТА №6")
    print("     ЕКСПОНЕНЦІАЛЬНЕ ЗГЛАДЖУВАННЯ ЧАСОВИХ РЯДІВ")
    print("-" * 80 + "\n")

    print("Оберіть джерело вхідних даних:")
    print("1 - Реальні дані (парсинг з сайту Minfin)")
    print("2 - Синтезовані дані (експоненціальний тренд + сезонність)")
    print("3 - Синтезовані дані (лінійний тренд + сезонність + АВ)")
    print("4 - Синтезовані дані (кубічний тренд + нормальний шум + АВ)")

    try:
        data_mode = int(input("\nВаш вибір (1-4): "))
    except:
        data_mode = 1
        print("Використовуємо режим за замовчуванням: 1")

    if data_mode == 1:
        print("\nРежим: РЕАЛЬНІ ДАНІ")
        data = parse_minfin_living_wage(use_backup=True, save_data=True)
    elif data_mode == 2:
        print("\nРежим: СИНТЕЗОВАНІ ДАНІ (експоненціальний)")
        data = synthesize_data(n=80, trend_type='exponential')
    elif data_mode == 3:
        print("\nРежим: СИНТЕЗОВАНІ ДАНІ (лінійний + АВ)")
        data = synthesize_data(n=80, trend_type='linear', add_anomalies=True)
    else:
        print("\nРежим: СИНТЕЗОВАНІ ДАНІ (кубічний + шум + АВ)")
        data = synthesize_data(n=100, trend_type='cubic')

    properties = analyze_time_series_properties(data)

    best_model, all_models, train, test = select_optimal_smoothing_model(data, properties=properties)

    forecast_periods = max(10, int(len(data) * 0.2))

    trained_model = train_and_forecast(
        data,
        best_model,
        forecast_periods=forecast_periods
    )

    if trained_model is None:
        print("\nПомилка навчання моделі. Програма завершена.")
        return

    kpi = evaluate_model_kpi(data, trained_model)

    visualize_results(
        data,
        trained_model,
        properties,
        best_model['name']
    )

    print("\n" + "-" * 80)
    print("ПРОГРАМА УСПІШНО ЗАВЕРШЕНА")

if __name__ == "__main__":
    main()