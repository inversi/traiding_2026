# binance_autotrader.py
# -*- coding: utf-8 -*-
# =============================================================
# Описание
# Этот скрипт — простой торговый бот для Binance (spot) на ccxt.
# Он читает параметры из файла .env и по 1-минутным свечам ищет лонг-setup
# (пробой максимума предыдущей свечи с фильтрами ATR и, при FAST_MODE, RSI на HTF).
# Обязательно начните с MODE=paper и только после проверки переключайте на live.
#
# Ключевые переменные .env (пример см. README/пояснение ниже):
#   MODE                 paper | live
#   MARKETS              Список торговых пар через запятую, например: PEPE/EUR,SHIB/EUR
#   TIMEFRAME            Основной ТФ для сигналов, например: 1m
#   LOOKBACK             Кол-во свечей для загрузки индикаторов и сигналов
#   RISK_PCT             Риск на сделку в долях депозита (0.007 = 0.7%).
#                        В текущей реализации не используется: размер позиции
#                        считается от фиксированной суммы входа (~10 EUR на сделку).
#                        Поле зарезервировано под альтернативный режим расчёта.
#   ATR_K                Множитель ATR для начального стоп-лосса (логический стоп)
#   ATR_PCT_MIN/MAX      Допустимая волатильность в долях цены (фильтр)
#   MAX_DAILY_DD_PCT     Лимит дневной просадки (в долях), при превышении — новые входы стоп
#   MAX_LOSSES_IN_ROW    Лимит серии убыточных сделок подряд (используется в on_tick)
#   FAST_MODE            Если true — включается MTF-фильтр тренда и объёма
#   FAST_HTF             Старший ТФ для фильтра (например 5m)
#   FAST_RSI_MIN         Минимальный RSI на HTF
#   FAST_MIN_VOL_SMA     Фильтр по объёму на 1m относительно SMA(объём)
#   EXCHANGE             Биржа (в этом примере только binance)
#   API_KEY / API_SECRET Ключи API для live-режима
#   TP_R_MULT            Тейк-профит в R (2.0 = 2R), если USE_TP=true
#   USE_TP               Ставить ли тейк-профит
#   USE_OCO_WHEN_AVAILABLE  Пытаться использовать OCO (тейк+стоп одним набором)
# =============================================================
# -------------------------------------------------------------
# Подробные комментарии:
# Ниже в коде добавлены пояснения к ключевым функциям, классам и
# участкам логики. Они должны помочь быстрее разобраться в том,
# как устроен бот, за что отвечает каждый блок и где безопасно
# вносить изменения под свои нужды.
# -------------------------------------------------------------
#
# -------------------------------------------------------------
# Краткое содержание основных компонентов
# -------------------------------------------------------------
# Функции индикаторов:
#   ema, sma       — экспоненциальная и простая скользящие средние
#   rsi            — индекс относительной силы (RSI) по Уайлдеру
#   atr            — средний истинный диапазон (ATR) по Уайлдеру
#
# Конфигурация и утилиты:
#   Config         — dataclass с параметрами бота (.env)
#   Position       — структура одной позиции (symbol, qty, entry, stop, tp)
#   load_config    — загрузка и разбор настроек из .env
#   parse_bool     — чтение булевых флагов из переменных окружения
#   fmt_float      — форматирование чисел для логов
#   format_ctx     — приведение контекста сигнала к строковому виду
#
# Логирование:
#   setup_logging  — настройка файловых логов (app/trades/errors)
#   log, log_info  — информационные сообщения
#   log_trade      — журнал входов/выходов и ордеров
#   log_error      — ошибки и исключения со стеком
#
# Обёртка биржи (Exchange):
#   fetch_ohlcv                — загрузка свечей OHLCV
#   balance_total_in_quote     — оценка всего спот-баланса в выбранной котируемой валюте
#   quote_free, base_free      — свободный баланс в котируемой/базовой валюте
#   avg_buy_price              — средняя цена покупок по символу
#   fetch_open_orders          — получение открытых ордеров
#   min_order_cost_quote       — минимальная стоимость ордера (minNotional)
#   price_step, min_price      — шаг и минимальная разрешённая цена
#   affordable                 — проверка, достаточно ли депозита для minNotional
#   round_qty                  — округление количества по лимитам биржи
#   create_market_buy/sell     — рыночные ордера
#   create_limit_sell          — лимитный ордер на продажу (TP)
#   create_stop_loss_limit     — стоп-лимитный ордер на продажу (SL)
#   create_oco_sell            — создание OCO (TP + SL)
#   cancel_order/_all_orders   — отмена ордеров
#
# Стратегия BreakoutWithATRAndRSI:
#   _signal                    — генерация сигнала на вход по пробою и фильтрам ATR/RSI/объёма
#   _position_size             — расчёт размера позиции от заданного риска
#   _place_orders              — выставление рыночного входа и защитных ордеров
#   _has_position_or_pending   — проверка существующих позиций и ожидающих ордеров
#   _compute_protective_prices — расчёт уровней SL/TP с учётом трейлинг-стопа
#   _reconcile_protective_orders — синхронизация биржевых SL/TP с логическими уровнями
#   bootstrap_existing_positions — подключение уже открытых вручную позиций
#   _cancel_position           — логическое закрытие позиции и принудительная продажа
#   on_tick                    — один цикл стратегии (сигналы, управление позициями)
#
# Точка входа:
#   main                       — запуск бота, префлайт-проверки и основной цикл
#
import os
import time
from datetime import datetime, date, timezone, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv

import logging
from logging.handlers import TimedRotatingFileHandler
import pathlib

# =========================
# Вспомогательные функции: индикаторы
# =========================
# Простейшие технические индикаторы.
# Все функции принимают pandas.Series и возвращают Series такой же длины.
# Важно: индикаторы не модифицируют входные данные, а создают новые столбцы.
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

# RSI по методу Уайлдера. Используется сглаженное скользящее среднее (EWMA)
# для усреднения положительных и отрицательных изменений цены.
# Возвращает значения от 0 до 100, где перекупленность/перепроданность
# зависят от выбранных порогов.
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    # Wilder's RSI
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

# ATR (Average True Range) по Уайлдеру. Считается на базе high/low/close и
# измеряет средний «истинный» диапазон свечей, то есть фактическую
# волатильность инструмента. Используется для расчёта стоп-лосса и фильтров.
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()  # Wilder's ATR

# =========================
# Модели данных
# =========================
# Конфигурация бота. Все параметры читаются из .env (см. функцию load_config).
# Здесь собраны настройки режима (paper/live), риск-менеджмент, фильтры,
# параметры индикаторов и подключения к бирже.
@dataclass
class Config:
    MODE: str
    MARKETS: List[str]
    TIMEFRAME: str
    LOOKBACK: int
    RISK_PCT: float
    ATR_K: float
    ATR_PCT_MIN: float
    ATR_PCT_MAX: float
    MAX_DAILY_DD_PCT: float
    MAX_LOSSES_IN_ROW: int
    VERBOSE: bool
    SLEEP_SEC: int
    FAST_MODE: bool
    FAST_HTF: str
    FAST_RSI_MIN: int
    FAST_ATR_K: float
    FAST_ATR_PCT_MIN: float
    FAST_ATR_PCT_MAX: float
    FAST_MIN_VOL_SMA: int
    EXCHANGE: str
    API_KEY: Optional[str]
    API_SECRET: Optional[str]
    TP_R_MULT: float  # take-profit multiple of risk (R); default 2.0
    USE_TP: bool
    USE_OCO_WHEN_AVAILABLE: bool
    FIXED_STOP_EUR: Optional[float] = None  # fixed stop-loss distance in quote currency

# Структура для хранения информации об одной позиции.
# В данном примере бот торгует только в лонг, поэтому side='long'.
# qty  — фактическое количество базовой валюты
# entry — цена входа, stop — уровень стоп-лосса, tp — уровень тейк-профита.
@dataclass
class Position:
    symbol: str
    side: str           # 'long' only in this example
    qty: float
    entry: float
    stop: float
    tp: Optional[float]

# =========================
# Вспомогательные утилиты
# =========================
# Утилита для чтения булевых флагов из переменных окружения.
# Поддерживаются значения вроде 'true', '1', 'yes' и т.п.
def parse_bool(v: str, default=False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

#
# Читает .env и формирует конфигурацию бота. Все поля имеют значения по умолчанию,
# чтобы удобно тестировать в paper-режиме. В live-режиме ключи API обязательны.
# Читает .env и формирует объект Config.
# Здесь задаются значения по умолчанию, чтобы можно было быстро запустить
# бота в paper-режиме без дополнительной настройки. В live-режиме
# обязательны ключи API_KEY / API_SECRET.
def load_config() -> Config:
    load_dotenv()
    MODE = os.getenv('MODE', 'paper').strip().lower()
    markets_env = os.getenv('MARKETS', 'BTC/EUR').replace(' ', '')
    MARKETS = [m for m in markets_env.split(',') if m]
    cfg = Config(
        MODE=MODE,
        MARKETS=MARKETS,
        TIMEFRAME=os.getenv('TIMEFRAME', '1m'),
        LOOKBACK=int(os.getenv('LOOKBACK', '600')),
        RISK_PCT=float(os.getenv('RISK_PCT', '0.005')),
        ATR_K=float(os.getenv('ATR_K', '3.0')),
        ATR_PCT_MIN=float(os.getenv('ATR_PCT_MIN', '0.002')),
        ATR_PCT_MAX=float(os.getenv('ATR_PCT_MAX', '0.08')),
        MAX_DAILY_DD_PCT=float(os.getenv('MAX_DAILY_DD_PCT', '0.02')),
        MAX_LOSSES_IN_ROW=int(os.getenv('MAX_LOSSES_IN_ROW', '3')),
        VERBOSE=parse_bool(os.getenv('VERBOSE', 'true'), True),
        SLEEP_SEC=int(os.getenv('SLEEP_SEC', '3')),
        FAST_MODE=parse_bool(os.getenv('FAST_MODE', 'false'), False),
        FAST_HTF=os.getenv('FAST_HTF', '5m'),
        FAST_RSI_MIN=int(os.getenv('FAST_RSI_MIN', '60')),
        FAST_ATR_K=float(os.getenv('FAST_ATR_K', '3.0')),
        FAST_ATR_PCT_MIN=float(os.getenv('FAST_ATR_PCT_MIN', '0.002')),
        FAST_ATR_PCT_MAX=float(os.getenv('FAST_ATR_PCT_MAX', '0.08')),
        FAST_MIN_VOL_SMA=int(os.getenv('FAST_MIN_VOL_SMA', '20')),
        EXCHANGE=os.getenv('EXCHANGE', 'binance').lower(),
        API_KEY=os.getenv('API_KEY') or None,
        API_SECRET=os.getenv('API_SECRET') or os.getenv('API_SECRET_KEY') or None,
        TP_R_MULT=float(os.getenv('TP_R_MULT', '2.0')),
        USE_TP=parse_bool(os.getenv('USE_TP', 'true'), True),
        USE_OCO_WHEN_AVAILABLE=parse_bool(os.getenv('USE_OCO_WHEN_AVAILABLE', 'true'), True),
        FIXED_STOP_EUR=float(os.getenv('FIXED_STOP_EUR', '0') or 0.0),
    )
    return cfg

# Вспомогательная функция для получения текущего времени в UTC в виде строки
# для логов, когда структурный логгер ещё не инициализирован.
def now_ts() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

# Обёртка над log_info: позволяет условно подавлять вывод в зависимости
# от флага verbose (используется для «болтливого» режима).
def log(msg: str, verbose=True):
    if verbose:
        log_info(msg)

# Форматирование float без экспоненциальной нотации, чтобы значения
# в логах было удобно читать (особенно для мелких цен/объёмов).
def fmt_float(x: float, digits: int = 8) -> str:
    try:
        return ("{0:." + str(digits) + "f}").format(float(x))
    except Exception:
        return str(x)

# Преобразование словаря контекста сигнала так, чтобы все float были
# отформатированы в строки с фиксированной точностью. Удобно для логов.
def format_ctx(ctx: dict, digits: int = 8) -> dict:
    """Возвращает копию словаря ctx, где все значения float отформатированы как строки с фиксированным числом знаков."""
    out = {}
    for k, v in ctx.items():
        if isinstance(v, float):
            out[k] = fmt_float(v, digits)
        else:
            out[k] = v
    return out

# Преобразует контекст в компактную строку вида key=value, key2=value2.
# Используется при логировании условий входа: видно, почему сигнал сработал
# или почему не прошёл фильтры.
def format_ctx_str(ctx: dict, digits: int = 8) -> str:
    """Удобное однострочное представление вида key=value с числами в фиксированном формате."""
    f = format_ctx(ctx, digits)
    parts = []
    for k, v in f.items():
        parts.append(f"{k}={v}")
    return "{" + ", ".join(parts) + "}"

# =========================
# Вспомогательные функции выбора рынков
# =========================
# Фильтр списка рынков по «доступности».
# Оставляет только те инструменты, по которым текущий депозит позволяет
# выставить минимальный ордер (minNotional). Это уменьшает количество
# бессмысленных попыток входа с недостаточным балансом.
def filter_markets_by_affordability(ex: "Exchange", markets: List[str], total_quote_balance: float, verbose: bool = True) -> List[str]:
    """
    Возвращает только те символы, по которым на аккаунте достаточно средств в котируемой валюте,
    чтобы выполнить минимальный ордер (minNotional, минимальная стоимость ордера).
    Проверка эквивалентна вызову `ex.affordable`.

    Параметры
    ----------
    ex : Exchange
        Инициализированная обёртка над биржей.
    markets : List[str]
        Кандидаты символов для торговли.
    total_quote_balance : float
        Баланс в котируемой валюте, используемый для проверки доступности (например, EUR),
        обычно тот же, что выводится в префлайт-логе в main().
    verbose : bool
        Если True, логирует, какие рынки оставлены или исключены.
    """
    tradable = []
    for sym in markets:
        try:
            ok, need = ex.affordable(sym, total_quote_balance)
        except Exception as e:
            if verbose:
                log(f"{sym}: affordability check failed with error: {e}")
            continue
        if ok:
            tradable.append(sym)
            if verbose:
                log(f"{sym}: включено в анализ (достаточно средств для мин. ордера)")
        else:
            if verbose:
                if need is None:
                    log(f"{sym}: исключено из анализа — нельзя определить мин. стоимость ордера")
                else:
                    log(f"{sym}: исключено из анализа — недостаточно средств для мин. ордера (нужно ≈ {need:.2f})")
    return tradable

# =========================
# Структурированное логирование
# =========================
LOGGER_APP = None
LOGGER_TRADES = None
LOGGER_ERRORS = None


# Инициализация структурированного логирования.
# Создаются три отдельных лог-файла:
#  - app.log    — общий ход работы бота
#  - trades.log — записи о входах/выходах и ордерах
#  - errors.log — стеки исключений и критические ошибки
# В дополнение к файлам информация дублируется в консоль.
def setup_logging(base_dir: str = 'logs'):
    global LOGGER_APP, LOGGER_TRADES, LOGGER_ERRORS
    log_dir = pathlib.Path(base_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Common formatter with UTC time
    formatter = logging.Formatter('[%(asctime)s UTC] %(message)s')
    formatter.converter = time.gmtime  # force UTC for strftime

    def make_handler(filename: str, level: int) -> TimedRotatingFileHandler:
        # Пишем в файл log_dir/filename, ротация по дням: каждые сутки создаётся новый файл
        # Формат имён по умолчанию: app.log.YYYY-MM-DD, trades.log.YYYY-MM-DD и т.п.
        h = TimedRotatingFileHandler(
            log_dir / filename,
            when="midnight",      # ротация в полночь по UTC (мы используем UTC в formatter)
            interval=1,
            backupCount=30,         # сколько дней логов хранить
            encoding="utf-8",
            utc=True                # чтобы ротация была привязана к UTC
        )
        h.setLevel(level)
        h.setFormatter(formatter)
        return h

    # App/general logger
    LOGGER_APP = logging.getLogger('app')
    LOGGER_APP.setLevel(logging.INFO)
    LOGGER_APP.handlers.clear()
    LOGGER_APP.addHandler(make_handler('app.log', logging.INFO))

    # Trades logger (executions, entries/exits, order placement results)
    LOGGER_TRADES = logging.getLogger('trades')
    LOGGER_TRADES.setLevel(logging.INFO)
    LOGGER_TRADES.handlers.clear()
    LOGGER_TRADES.addHandler(make_handler('trades.log', logging.INFO))

    # Errors logger
    LOGGER_ERRORS = logging.getLogger('errors')
    LOGGER_ERRORS.setLevel(logging.ERROR)
    LOGGER_ERRORS.handlers.clear()
    LOGGER_ERRORS.addHandler(make_handler('errors.log', logging.ERROR))

    # Console handler for quick view
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    LOGGER_APP.addHandler(console)
    LOGGER_TRADES.addHandler(console)
    LOGGER_ERRORS.addHandler(console)


def log_info(msg: str):
    if LOGGER_APP is not None:
        LOGGER_APP.info(msg)
    else:
        print(f"[{now_ts()}] {msg}")


def log_trade(msg: str):
    if LOGGER_TRADES is not None:
        LOGGER_TRADES.info(msg)
    else:
        print(f"[{now_ts()}] {msg}")


def log_error(msg: str, exc: Exception = None):
    if LOGGER_ERRORS is not None:
        if exc is not None:
            LOGGER_ERRORS.error(msg + f" | exception={exc}", exc_info=True)
        else:
            LOGGER_ERRORS.error(msg)
    else:
        print(f"[{now_ts()}] ERROR: {msg}")

# =========================
# Обёртка биржи (ccxt)
# =========================
# Обёртка над ccxt для конкретной биржи (здесь — Binance spot).
# Содержит методы для:
#  - загрузки свечей
#  - оценки баланса в котируемой валюте
#  - вычисления minNotional, шага цены и т.д.
#  - создания/отмены рыночных, лимитных и OCO-ордеров
# Этот слой абстракции позволяет, при желании, позже адаптировать код
# под другую биржу, изменив реализацию только этого класса.
class Exchange:
    def __init__(self, cfg: Config):
        # Инициализация клиента ccxt для Binance spot. Загружаем рынки и
        # проверяем поддержку OCO (на некоторых аккаунтах/регионах может отличаться).
        if cfg.EXCHANGE != 'binance':
            raise ValueError('Only binance is supported in this sample.')
        self.cfg = cfg
        self.ccxt = ccxt.binance({
            'apiKey': cfg.API_KEY or '',
            'secret': cfg.API_SECRET or '',
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        # Ensure unified symbols like 'PEPE/EUR' are loaded
        self.markets = self.ccxt.load_markets()
        self.has_oco = bool(getattr(self.ccxt, 'has', {}).get('createOrderOCO', False))

    # Загружает OHLCV-данные и превращает их в DataFrame с удобными именами
    # колонок. Именно с этим форматом далее работает стратегия.
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        # Загружаем OHLCV и приводим к DataFrame с колонками ts/open/high/low/close/volume.
        ohlcv = self.ccxt.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df

    # Оценка общего спот-баланса в выбранной котируемой валюте (по умолчанию EUR).
    # Перебираются все активы кошелька и для каждого ищется путь конверсии
    # в quote (прямо, через USDT или через BTC). Нельзя использовать для
    # точной финансовой отчётности, но достаточно для risk-менеджмента.
    def balance_total_in_quote(self, quote: str = 'EUR') -> float:
        """Оценивает общий спот-баланс в выбранной котируемой валюте.

        Стратегия пересчёта:
          1) Прямая пара ASSET/QUOTE (или QUOTE/ASSET с обращением курса)
          2) Через USDT: ASSET/USDT и USDT/QUOTE (в любом направлении)
          3) Через BTC:  ASSET/BTC и BTC/QUOTE (в любом направлении)

        Если ни один путь не найден, актив пропускается при оценке.
        Используются балансы 'total' (free + used) спотового кошелька.
        """
        def _pair_last(base: str, q: str) -> Optional[float]:
            # Try BASE/QUOTE, else QUOTE/BASE with inversion
            pair = f"{base}/{q}"
            try:
                t = self.ccxt.fetch_ticker(pair)
                px = float(t.get('last') or t.get('close') or ((t.get('bid') or 0)+(t.get('ask') or 0))/2 or 0.0)
                if px > 0:
                    return px
            except Exception:
                pass
            # Try reversed
            pair_rev = f"{q}/{base}"
            try:
                t = self.ccxt.fetch_ticker(pair_rev)
                px = float(t.get('last') or t.get('close') or ((t.get('bid') or 0)+(t.get('ask') or 0))/2 or 0.0)
                if px > 0:
                    return 1.0 / px if px != 0 else None
            except Exception:
                pass
            return None

        def _asset_in_quote(asset: str, q: str) -> Optional[float]:
            if asset == q:
                return 1.0
            # 1) direct
            px = _pair_last(asset, q)
            if px is not None:
                return px
            # 2) via USDT
            via = 'USDT'
            px1 = _pair_last(asset, via)
            px2 = _pair_last(via, q)
            if (px1 is not None) and (px2 is not None):
                return px1 * px2
            # 3) via BTC
            via = 'BTC'
            px1 = _pair_last(asset, via)
            px2 = _pair_last(via, q)
            if (px1 is not None) and (px2 is not None):
                return px1 * px2
            return None

        try:
            bal = self.ccxt.fetch_balance()
        except Exception:
            return 0.0

        totals = bal.get('total') or {}
        total_value = 0.0
        for asset, amt in totals.items():
            try:
                amount = float(amt or 0.0)
            except Exception:
                amount = 0.0
            if amount == 0.0:
                continue
            price_q = _asset_in_quote(asset, quote)
            if price_q is None:
                # couldn't value this asset in the requested quote
                continue
            total_value += amount * price_q
        return float(total_value)

    # Оценка цены актива в BTC. Нужна для определения «пыли», которую можно
    # конвертировать в BNB через Binance Small Amount Exchange.
    def price_in_btc(self, asset: str) -> Optional[float]:
        asset = asset.upper()
        if asset == 'BTC':
            return 1.0
        pair_direct = f"{asset}/BTC"
        pair_reverse = f"BTC/{asset}"
        for pair, invert in ((pair_direct, False), (pair_reverse, True)):
            try:
                t = self.ccxt.fetch_ticker(pair)
                px = float(t.get('last') or t.get('close') or ((t.get('bid') or 0) + (t.get('ask') or 0)) / 2 or 0.0)
                if px > 0:
                    return (1.0 / px) if invert else px
            except Exception:
                continue
        return None

    # Возвращает стоимость указанного количества актива в BTC (или None, если цену не удалось получить).
    def value_in_btc(self, asset: str, amount: float) -> Optional[float]:
        px = self.price_in_btc(asset)
        if px is None:
            return None
        try:
            return float(amount) * float(px)
        except Exception:
            return None

    # Подбор активов, пригодных для Binance Small Amount Exchange (dust to BNB).
    # Фильтрует свободные остатки, которые оцениваются дешевле max_value_btc и не входят в список исключений.
    def find_small_balances_for_bnb(self, max_value_btc: float = 0.001, min_free: float = 1e-8, exclude: Optional[List[str]] = None) -> List[str]:
        bal = self.ccxt.fetch_balance()
        free_bal = bal.get('free') or {}
        exclude_set = set(a.upper() for a in (exclude or ['BNB', 'BTC', 'USDT', 'BUSD', 'FDUSD', 'USDC', 'EUR']))
        candidates: List[str] = []
        for asset, amt in free_bal.items():
            try:
                amount = float(amt or 0.0)
            except Exception:
                continue
            if amount <= min_free:
                continue
            asset_u = asset.upper()
            if asset_u in exclude_set:
                continue
            value_btc = self.value_in_btc(asset_u, amount)
            if value_btc is None:
                continue
            if value_btc <= max_value_btc:
                candidates.append(asset_u)
        return candidates

    # Выполняет конверсию «пыли» в BNB через /sapi/v1/asset/dust.
    # По умолчанию собирает кандидатов автоматически, но можно передать явный список assets.
    def convert_small_balances_to_bnb(self, assets: Optional[List[str]] = None, max_value_btc: float = 0.001) -> dict:
        if self.cfg.MODE != 'live':
            raise RuntimeError("Dust conversion доступен только в live-режиме.")
        if not hasattr(self.ccxt, 'sapiPostAssetDust'):
            raise RuntimeError("ccxt/binance не поддерживает sapiPostAssetDust на этом аккаунте.")

        assets = assets or self.find_small_balances_for_bnb(max_value_btc=max_value_btc)
        assets = [a for a in assets if a and a.upper() != 'BNB']
        if not assets:
            return {}
        try:
            return self.ccxt.sapiPostAssetDust({'asset': assets})
        except Exception as e:
            log_error("Dust conversion failed", e)
            return {}

    # Свободный (не зарезервированный ордерами) баланс в выбранной
    # котируемой валюте на споте.
    def quote_free(self, quote: str = 'EUR') -> float:
        bal = self.ccxt.fetch_balance()
        try:
            return float(bal.get('free', {}).get(quote, 0.0) or 0.0)
        except Exception:
            return 0.0

    # Максимальное количество базовой валюты, которое можно купить
    # на весь доступный free-баланс quote с небольшим запасом safety,
    # чтобы не упереться в ошибку недостатка средств.
    def max_buy_qty(self, symbol: str, safety: float = 0.995) -> float:
        quote = symbol.split('/')[1]
        free_quote = self.quote_free(quote)
        px = self.last_price(symbol)
        if px <= 0 or free_quote <= 0:
            return 0.0
        raw = (free_quote / px) * max(0.0, min(1.0, safety))
        # If the calculated raw amount is non-positive, skip early
        if raw <= 0:
            return 0.0
        # Some markets (e.g. LINK/EUR) require the amount to be >= precision step.
        # amount_to_precision may raise InvalidOrder if raw is too small; in that case
        # we just report that we cannot buy anything meaningful.
        try:
            return float(self.ccxt.amount_to_precision(symbol, raw))
        except Exception:
            log(f"{symbol}: max_buy_qty amount_to_precision rejected raw={raw:.12g} — returning 0", True)
            return 0.0

    # Свободный остаток базовой валюты по символу (например, PEPE для PEPE/EUR).
    # Используется при bootstrap'е для обнаружения уже существующих ручных позиций.
    def base_free(self, symbol: str) -> float:
        """Свободный остаток базовой валюты по символу (например, PEPE для PEPE/EUR)."""
        base = symbol.split('/')[0]
        bal = self.ccxt.fetch_balance()
        try:
            return float(bal.get('free', {}).get(base, 0.0) or 0.0)
        except Exception:
            return 0.0

    # Приблизительная средняя цена покупок по символу за заданный период.
    # Нужна для того, чтобы при bootstrap'е задать разумный уровень entry,
    # если позиция была открыта вручную до запуска бота.
    def avg_buy_price(self, symbol: str, lookback_days: int = 30) -> Optional[float]:
        """Оцениваем среднюю цену покупок по символу за последние N дней (spot).
        Возвращает None, если нет данных о трейдах. Если у Binance нет полной истории в рамках лимита — это приблизительная оценка.
        """
        try:
            since_ms = int((datetime.now(timezone.utc).timestamp() - lookback_days * 86400) * 1000)
            trades = self.ccxt.fetch_my_trades(symbol, since=since_ms)
            buys = [t for t in trades if str(t.get('side')) == 'buy']
            if not buys:
                return None
            cost = 0.0
            amount = 0.0
            for t in buys:
                px = float(t.get('price') or 0.0)
                qty = float(t.get('amount') or 0.0)
                if px > 0 and qty > 0:
                    cost += px * qty
                    amount += qty
            if amount <= 0:
                return None
            return cost / amount
        except Exception:
            return None

    # Обёртка над fetch_open_orders ccxt. Возвращает список всех открытых
    # ордеров (по символу или по всем рынкам), чтобы стратегия могла
    # синхронизировать внутреннее состояние с тем, что реально висит на бирже.
    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[dict]:
        """Возвращает список открытых ордеров (spot). Если symbol=None — по всем рынкам."""
        try:
            return self.ccxt.fetch_open_orders(symbol)
        except Exception:
            return []

    def market_info(self, symbol: str) -> dict:
        return self.markets[symbol]

    def last_price(self, symbol: str) -> float:
        """Последняя цена по тикеру (last)."""
        t = self.ccxt.fetch_ticker(symbol)
        return float(t.get('last') or t.get('close') or t.get('bid') or 0.0)

    # Возвращает оценку минимальной стоимости ордера (minNotional) в котируемой валюте.
    # Сначала пытается взять limits.cost.min из описания рынка, при отсутствии —
    # оценивает как min_amount * last_price. Если информации нет, вернёт None.
    def min_order_cost_quote(self, symbol: str, fallback_price: Optional[float] = None) -> Optional[float]:
        """Минимальная стоимость ордера в котируемой валюте (quote), если доступно.
        Пытается взять limits.cost.min, иначе оценивает как min_amount * last_price.
        Возвращает None, если оценить нельзя.
        """
        m = self.market_info(symbol)
        # 1) Пробуем прямой min notional из limits.cost.min
        cost_limits = m.get('limits', {}).get('cost') or {}
        min_cost = cost_limits.get('min')
        if min_cost is not None:
            try:
                return float(min_cost)
            except Exception:
                pass
        # 2) Если нет — пробуем amount.min * last_price
        amt_limits = m.get('limits', {}).get('amount') or {}
        min_amount = amt_limits.get('min')
        if min_amount is None:
            # иногда min находится в m['precision'] — тогда округлим 1e-precision
            prec = m.get('precision', {}).get('amount')
            if isinstance(prec, int) and prec >= 0:
                min_amount = 10 ** (-prec)
        if min_amount is None:
            return None
        price = fallback_price if fallback_price is not None else self.last_price(symbol)
        if price <= 0:
            return None
        try:
            return float(min_amount) * float(price)
        except Exception:
            return None

    # Шаг цены (tickSize) для инструмента. На Binance сначала читается
    # PRICE_FILTER.tickSize, при отсутствии — шаг оценивается по precision.
    def price_step(self, symbol: str) -> float:
        """Возвращает минимальный шаг цены (tickSize) для символа, предпочитая Binance PRICE_FILTER.tickSize."""
        m = self.market_info(symbol)
        # Try exchange-specific filters first
        try:
            for f in m.get('info', {}).get('filters', []):
                if f.get('filterType') == 'PRICE_FILTER':
                    ts = float(f.get('tickSize') or 0)
                    if ts > 0:
                        return ts
        except Exception:
            pass
        # Fallback to precision-based step
        prec = (m.get('precision') or {}).get('price')
        if isinstance(prec, int) and prec >= 0:
            try:
                return float(10 ** (-prec))
            except Exception:
                pass
        # Last resort
        lim_min = (m.get('limits', {}).get('price') or {}).get('min')
        try:
            return float(lim_min or 0.0)
        except Exception:
            return 0.0

    # Минимально допустимая цена для инструмента согласно правилам биржи.
    # Нужна при расчёте стоп-лимит и OCO ордеров, чтобы не отправлять
    # на биржу цены ниже разрешённого уровня.
    def min_price(self, symbol: str) -> float:
        """Возвращает минимально допустимую цену для символа, если доступна (иначе 0)."""
        m = self.market_info(symbol)
        try:
            for f in m.get('info', {}).get('filters', []):
                if f.get('filterType') == 'PRICE_FILTER':
                    mp = float(f.get('minPrice') or 0)
                    return mp
        except Exception:
            pass
        lim_min = (m.get('limits', {}).get('price') or {}).get('min')
        try:
            return float(lim_min or 0.0)
        except Exception:
            return 0.0

    # Определение допустимых границ цены согласно фильтру PERCENT_PRICE_BY_SIDE.
    # Binance проверяет цену относительно средневзвешенной цены за период avgPriceMins,
    # поэтому используем weightedAvgPrice из тикера, а не только last.
    def _percent_price_bounds(self, symbol: str, side: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Возвращает (min_allowed, max_allowed, ref_price) для PERCENT_PRICE_BY_SIDE, если доступно."""
        side_l = str(side).lower()
        m = self.market_info(symbol)
        filters = (m.get('info') or {}).get('filters', []) or []

        # Ищем мультипликаторы для нужной стороны (SELL -> ask*, BUY -> bid*).
        mult_down = None
        mult_up = None
        for f in filters:
            try:
                if f.get('filterType') == 'PERCENT_PRICE_BY_SIDE':
                    if side_l == 'sell':
                        mult_down = float(f.get('askMultiplierDown')) if f.get('askMultiplierDown') is not None else None
                        mult_up = float(f.get('askMultiplierUp')) if f.get('askMultiplierUp') is not None else None
                    else:
                        mult_down = float(f.get('bidMultiplierDown')) if f.get('bidMultiplierDown') is not None else None
                        mult_up = float(f.get('bidMultiplierUp')) if f.get('bidMultiplierUp') is not None else None
                    break
            except Exception:
                continue

        # Тикер нужен, чтобы взять weightedAvgPrice (ближе к биржевому расчёту avgPriceMins).
        ref_price = None
        try:
            t = self.ccxt.fetch_ticker(symbol)
        except Exception:
            t = None

        if t:
            info = t.get('info') or {}
            candidates = [
                info.get('weightedAvgPrice'),
                info.get('weightedAveragePrice'),
                t.get('average'),
                t.get('vwap'),
                t.get('last'),
                t.get('close'),
                t.get('bid'),
                t.get('ask'),
            ]
            for c in candidates:
                try:
                    if c is not None:
                        ref_price = float(c)
                        if ref_price > 0:
                            break
                except Exception:
                    continue

        if (ref_price is None) or (ref_price <= 0):
            # Последний шанс — last_price (может вызвать доп. запрос, но лучше, чем None).
            try:
                ref_price = self.last_price(symbol)
            except Exception:
                ref_price = None

        if (ref_price is None) or (ref_price <= 0) or (mult_down is None):
            return None, None, ref_price

        min_allowed = ref_price * mult_down
        max_allowed = (ref_price * mult_up) if (mult_up is not None) else None
        return min_allowed, max_allowed, ref_price

    # Проверка, достаточно ли средств в котируемой валюте для выполнения
    # минимального ордера по символу. Используется как при префлайт-проверке,
    # так и при фильтрации рынков.
    def affordable(self, symbol: str, quote_balance: float) -> Tuple[bool, Optional[float]]:
        """Проверка: достаточно ли средств в котируемой валюте для минимального ордера."""
        min_cost = self.min_order_cost_quote(symbol)
        if min_cost is None:
            return False, None
        return (quote_balance >= min_cost), float(min_cost)

    # Округление количества базовой валюты до требований биржи.
    # Учитывает precision (кол-во знаков), минимальный размер ордера и
    # особый случай целочисленных количеств (precision == 0), чтобы избежать
    # ошибок InvalidOrder со стороны ccxt/Binance.
    # Если после всех проверок количество получается ниже допустимого, возвратит 0.0.
    def round_qty(self, symbol: str, qty: float) -> float:
        """Округляет количество до точности биржи и учитывает минимально допустимый объём.

        Избегает ошибок ccxt InvalidOrder, проверяя случай precision == 0 (целочисленные объёмы)
        и явно заданный минимальный размер. Если после округления объём получается ниже
        минимально разрешённого, возвращает 0.0, чтобы вызывающий код мог пропустить сделку.
        """
        info = self.market_info(symbol)
        limits_amount = (info.get('limits') or {}).get('amount') or {}
        min_amount = limits_amount.get('min')
        prec = (info.get('precision') or {}).get('amount')

        # If the market requires integer amounts (precision == 0) and qty < 1, skip early
        try:
            if isinstance(prec, int) and prec == 0 and qty < 1:
                log(f"{symbol}: qty {qty:.8f} < 1 and precision=0 (integer size required) — skip", True)
                return 0.0
        except Exception:
            pass

        # First, try to round to exchange precision using ccxt helper
        try:
            qty_rounded = float(self.ccxt.amount_to_precision(symbol, qty))
        except Exception:
            # If helper complains (e.g., qty below precision step), skip gracefully
            log(f"{symbol}: amount_to_precision rejected qty={qty:.12g} — skip", True)
            return 0.0

        # Enforce explicit minimum amount if provided by the market
        if min_amount is not None:
            try:
                min_amount = float(min_amount)
            except Exception:
                min_amount = None
        if (min_amount is not None) and (qty_rounded < min_amount):
            log(f"{symbol}: qty_rounded {qty_rounded:.8f} < min_amount {min_amount:.8f} — skip", True)
            return 0.0

        return float(qty_rounded)

    # Рынок-покупка по текущей цене. Вся логика проверки размера/баланса
    # выполняется в стратегии до вызова этого метода.
    def create_market_buy(self, symbol: str, amount: float) -> dict:
        return self.ccxt.create_order(symbol, 'market', 'buy', amount)

    # Рынок-продажа по текущей цене. Используется при принудительном закрытии
    # позиции, чтобы продать весь доступный объём базовой валюты.
    def create_market_sell(self, symbol: str, amount: float) -> dict:
        amount_p = float(self.ccxt.amount_to_precision(symbol, amount))
        return self.ccxt.create_order(symbol, 'market', 'sell', amount_p)

    # Лимитный ордер на продажу (обычно для TP). Все значения приводятся
    # к нужной точности через ccxt.*_to_precision. Для Binance дополнительно
    # учитываем фильтр PERCENT_PRICE_BY_SIDE, чтобы не выходить за допустимый
    # диапазон цен относительно текущего рынка.
    def create_limit_sell(self, symbol: str, amount: float, price: float, params: dict=None) -> dict:
        # Binance валидирует цены относительно weightedAvgPrice*multiplier,
        # поэтому используем границы из фильтра PERCENT_PRICE_BY_SIDE.
        min_allowed, max_allowed, ref_price = self._percent_price_bounds(symbol, 'sell')
        if (min_allowed is not None) and (ref_price is not None) and ref_price > 0:
            try:
                price = max(price, min_allowed)
                if max_allowed is not None:
                    price = min(price, max_allowed)
            except Exception:
                # Если что-то пошло не так при обрезке — оставляем исходную цену,
                # пусть ccxt/биржа сообщит об ошибке.
                pass

        amount_p = float(self.ccxt.amount_to_precision(symbol, amount))
        price_p = float(self.ccxt.price_to_precision(symbol, price))
        return self.ccxt.create_order(symbol, 'limit', 'sell', amount_p, price_p, params or {})

    # Стоп-лимитный ордер на продажу для Binance spot. Биржа требует
    # одновременно указать stopPrice (триггер) и price (лимитная цена).
    def create_stop_loss_limit(self, symbol: str, amount: float, stop_price: float, limit_price: float) -> dict:
        # Binance spot stop-loss-limit:
        #  - type 'stop_loss_limit'
        #  - price (limit price) + param stopPrice
        #  - обе цены должны проходить PRICE_FILTER и PERCENT_PRICE_BY_SIDE
        min_allowed, max_allowed, ref_price = self._percent_price_bounds(symbol, 'sell')
        if (min_allowed is not None) and (ref_price is not None) and ref_price > 0:
            try:
                # Для SELL-ордеров Binance требует:
                #   stop_price, limit_price >= ref_price * askMultiplierDown
                #   stop_price, limit_price <= ref_price * askMultiplierUp
                stop_price = max(stop_price, min_allowed)
                limit_price = max(limit_price, min_allowed)
                if max_allowed is not None:
                    stop_price = min(stop_price, max_allowed)
                    limit_price = min(limit_price, max_allowed)
            except Exception:
                # В случае любой ошибки просто не трогаем исходные уровни
                pass

        amount_p = float(self.ccxt.amount_to_precision(symbol, amount))
        stop_p = float(self.ccxt.price_to_precision(symbol, stop_price))
        limit_p = float(self.ccxt.price_to_precision(symbol, limit_price))
        params = {
            'stopPrice': stop_p,
            'timeInForce': 'GTC',
        }
        return self.ccxt.create_order(symbol, 'stop_loss_limit', 'sell', amount_p, limit_p, params)

    # Создание OCO-ордера (One-Cancels-the-Other) через SAPI Binance.
    # Состоит из связки лимитного TP и стоп-лимит ордера. При исполнении
    # одного из них второй автоматически отменяется биржей.
    def create_oco_sell(self, symbol: str, amount: float, take_profit_price: float, stop_price: float, stop_limit_price: float) -> dict:
        # Binance OCO требует:
        #  - корректные precision
        #  - выполнение PRICE_FILTER и PERCENT_PRICE_BY_SIDE для всех цен (TP/SL)
        min_allowed, max_allowed, ref_price = self._percent_price_bounds(symbol, 'sell')
        if (min_allowed is not None) and (ref_price is not None) and ref_price > 0:
            try:
                # Ограничиваем все цены снизу/сверху допустимым диапазоном.
                take_profit_price = max(take_profit_price, min_allowed)
                stop_price = max(stop_price, min_allowed)
                stop_limit_price = max(stop_limit_price, min_allowed)
                if max_allowed is not None:
                    take_profit_price = min(take_profit_price, max_allowed)
                    stop_price = min(stop_price, max_allowed)
                    stop_limit_price = min(stop_limit_price, max_allowed)
            except Exception:
                pass

        symbol_id = self.ccxt.market_id(symbol) if hasattr(self.ccxt, 'market_id') else symbol.replace('/', '')
        params = {
            'symbol': symbol_id,
            'side': 'SELL',
            'quantity': self.ccxt.amount_to_precision(symbol, amount),
            'price': self.ccxt.price_to_precision(symbol, take_profit_price),
            'stopPrice': self.ccxt.price_to_precision(symbol, stop_price),
            'stopLimitPrice': self.ccxt.price_to_precision(symbol, stop_limit_price),
            'stopLimitTimeInForce': 'GTC',
        }
        return self.ccxt.sapiPostOrderOco(params)

    # Отмена одиночного ордера по его id. Обёртка над ccxt.cancel_order,
    # которая глушит исключения и всегда возвращает словарь.
    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> dict:
        """Отменяет одиночный ордер по идентификатору (опционально с указанием символа)."""
        try:
            return self.ccxt.cancel_order(order_id, symbol)
        except Exception:
            return {}

    # Массовая отмена всех открытых ордеров (опционально только по одному символу).
    # Проходит по списку открытых ордеров и пытается отменить каждый.
    def cancel_all_orders(self, symbol: Optional[str] = None) -> None:
        """Отменяет все открытые ордера, опционально только по одному символу."""
        try:
            open_orders = self.fetch_open_orders(symbol)
        except Exception:
            open_orders = []
        for o in open_orders:
            try:
                oid = o.get('id') or o.get('orderId') or (o.get('info') or {}).get('orderId')
                if oid:
                    self.cancel_order(str(oid), symbol)
            except Exception:
                continue

# =========================
# Сбор истории ордеров и сделок
# =========================
# Вспомогательная функция, которую можно вызвать вручную (например, из REPL
# или отдельного скрипта), чтобы выгрузить историю ордеров и сделок за
# последний месяц в CSV-файлы для последующего анализа.
#
# ВАЖНО: функция НЕ вызывается автоматически внутри main() или стратегии,
# чтобы не замедлять работу бота. Её нужно вызывать явно:
#
#   from bot import Exchange, load_config, collect_history_last_month
#   cfg = load_config()
#   ex = Exchange(cfg)
#   collect_history_last_month(ex, symbols=cfg.MARKETS)
#
def collect_history_last_month(ex: Exchange, symbols: Optional[List[str]] = None, days: int = 30, out_dir: str = 'logs/history') -> None:
    """
    Выгружает историю сделок и ордеров за последние `days` дней (по умолчанию ~месяц)
    и сохраняет их в CSV-файлы в каталоге `out_dir`.

    Параметры
    ----------
    ex : Exchange
        Инициализированная обёртка биржи.
    symbols : Optional[List[str]]
        Список символов для выгрузки. Если None — биржа вернёт историю по всем доступным символам.
        (Поддерживается не всеми биржами; для Binance spot работает.)
    days : int
        Количество дней истории, по умолчанию 30.
    out_dir : str
        Папка для сохранения CSV-файлов.
    """
    os.makedirs(out_dir, exist_ok=True)
    since_dt = datetime.now(timezone.utc) - timedelta(days=days)
    since_ms = int(since_dt.timestamp() * 1000)

    # Сбор сделок (trade history)
    try:
        if symbols:
            all_trades = []
            for sym in symbols:
                try:
                    ts = ex.ccxt.fetch_my_trades(sym, since=since_ms)
                    for t in ts:
                        if 'symbol' not in t:
                            t['symbol'] = sym
                    all_trades.extend(ts)
                except Exception as e:
                    log(f"{sym}: не удалось получить историю сделок: {e}", True)
            trades = all_trades
        else:
            trades = ex.ccxt.fetch_my_trades(since=since_ms)
    except Exception as e:
        log_error("Не удалось получить историю сделок за период", e)
        trades = []

    if trades:
        try:
            df_trades = pd.DataFrame(trades)
            trades_path = os.path.join(out_dir, f"trades_last_{days}d.csv")
            df_trades.to_csv(trades_path, index=False)
            log(f"История сделок сохранена в {trades_path} (строк: {len(df_trades)})")
        except Exception as e:
            log_error("Ошибка при сохранении истории сделок в CSV", e)
    else:
        log("История сделок за указанный период пуста или недоступна.", True)

    # Сбор ордеров (order history)
    try:
        if symbols:
            all_orders = []
            for sym in symbols:
                try:
                    osym = ex.ccxt.fetch_orders(sym, since=since_ms)
                    for o in osym:
                        if 'symbol' not in o:
                            o['symbol'] = sym
                    all_orders.extend(osym)
                except Exception as e:
                    log(f"{sym}: не удалось получить историю ордеров: {e}", True)
            orders = all_orders
        else:
            orders = ex.ccxt.fetch_orders(since=since_ms)
    except Exception as e:
        log_error("Не удалось получить историю ордеров за период", e)
        orders = []

    if orders:
        try:
            df_orders = pd.DataFrame(orders)
            orders_path = os.path.join(out_dir, f"orders_last_{days}d.csv")
            df_orders.to_csv(orders_path, index=False)
            log(f"История ордеров сохранена в {orders_path} (строк: {len(df_orders)})")
        except Exception as e:
            log_error("Ошибка при сохранении истории ордеров в CSV", e)
    else:
        log("История ордеров за указанный период пуста или недоступна.", True)


# =========================
# Логика стратегии
# =========================
# Основная логика стратегии.
# Стратегия ищет лонг-входы по пробою максимума предыдущей свечи с фильтрами
# по ATR (волатильность), RSI на старшем таймфрейме (если включён FAST_MODE)
# и объёму. Управляет позициями, стопами, тейк-профитами и дневными лимитами
# по просадке и серии убыточных сделок.
class BreakoutWithATRAndRSI:
    # Проверяет, есть ли уже позиция или открытые ордера по символу.
    # Нужна, чтобы не открывать дубликаты позиций и не накладывать
    # несколько независимых стопов/тейков на один и тот же инструмент.
    def _has_position_or_pending(self, symbol: str) -> bool:
        """True если уже есть позиция (в трекере или по балансу) или есть открытые ордера по символу."""
        # Уже учтённая позиция
        if symbol in self.positions and self.positions[symbol].qty > 0:
            return True
        # Остаток базовой монеты (например, при рестарте бота до bootstrap)
        try:
            qty = self.ex.base_free(symbol)
            if qty > 0:
                last = self.ex.last_price(symbol)
                min_cost = self.ex.min_order_cost_quote(symbol, fallback_price=last)
                # treat as a position only if the residual is tradable (>= minNotional)
                if (min_cost is None) or (qty * last >= float(min_cost)):
                    return True
        except Exception:
            pass
        # Ожидающие ордера на символ (buy/sell/OCO legs)
        try:
            oo = self.ex.fetch_open_orders(symbol)
            if oo:
                # Оценка суммарной стоимости открытых ордеров (для диагностики и логов).
                # Для лимитных/стоп-ордеров берём price или stopPrice, при отсутствии — last_price.
                if self.cfg.VERBOSE:
                    try:
                        last_px = self.ex.last_price(symbol)
                    except Exception:
                        last_px = 0.0
                    for o in oo:
                        try:
                            info = o.get('info') or {}
                            side = str(o.get('side') or info.get('side') or '').lower()
                            amount = o.get('amount')
                            # Если amount=None, пробуем remaining / origQty из info
                            if amount is None:
                                amount = o.get('remaining') or info.get('origQty')
                            amount_f = float(amount or 0.0)

                            price = o.get('price') or info.get('price') or info.get('stopPrice') or last_px
                            price_f = float(price or 0.0)

                            est_cost = amount_f * price_f
                            log(
                                f"{symbol}: open order side={side} amount={fmt_float(amount_f,8)} "
                                f"price≈{fmt_float(price_f,8)} est_cost≈{fmt_float(est_cost,2)}",
                                True
                            )
                        except Exception:
                            continue
                return True
        except Exception:
            pass
        return False

    # Расчёт уровней защитных ордеров (SL и TP) относительно цены входа и текущего логического стопа.
    # В этой версии стоп берётся из текущего trailing stop (logical_stop), а TP вычисляется как entry + TP_R_MULT * (entry - stop).
    def _compute_protective_prices(self, symbol: str, entry: float, logical_stop: float, last_price: float) -> Tuple[float, float, Optional[float]]:
        """Расчёт биржевых уровней защитных ордеров (stop/stop_limit/tp) по логическому стопу.

        Возвращает кортеж (stop_price, stop_limit_price, tp_price или None) на основе текущего логического стопа.

        Все уровни корректируются под требования биржи (округление и строгие неравенства):
        - entry         — цена входа
        - logical_stop  — текущий «логический» уровень стопа (например, подтянутый трейлингом)
        - last_price    — текущая рыночная цена
        """
        # Базовый уровень стопа берём из логического стопа (trailing), а не пересчитываем от ATR каждый раз.
        stop = float(logical_stop)

        # Если по какой-то причине стоп оказался выше текущей цены, немного отодвинем его ниже last_price.
        step = max(self.ex.price_step(symbol), 0.0) or (max(last_price, 1e-12) * 0.001)
        minp = self.ex.min_price(symbol)

        stop_price = min(stop, last_price - step)
        stop_limit_price = max(minp, stop_price - step)

        # Тейк-профит считаем как entry + TP_R_MULT * R, где R = entry - stop.
        tp_price = None
        if self.cfg.USE_TP:
            risk_per_unit = max(0.0, float(entry) - stop)
            if risk_per_unit > 0:
                raw_tp = float(entry) + self.cfg.TP_R_MULT * risk_per_unit
                tp_price = max(raw_tp, last_price + step, stop_price + step)

        # Округляем уровни по требованиям биржи
        stop_price = float(self.ex.ccxt.price_to_precision(symbol, stop_price))
        stop_limit_price = float(self.ex.ccxt.price_to_precision(symbol, stop_limit_price))
        if tp_price is not None:
            tp_price = float(self.ex.ccxt.price_to_precision(symbol, tp_price))

        # Гарантируем строгие неравенства после округления, не опускаясь ниже минимальной цены.
        if tp_price is not None and not (tp_price > last_price and stop_price < last_price and stop_limit_price < stop_price):
            adj_stop = max(minp, stop_price - step)
            adj_sllim = max(minp, stop_limit_price - step)
            stop_price = float(self.ex.ccxt.price_to_precision(symbol, adj_stop))
            stop_limit_price = float(self.ex.ccxt.price_to_precision(symbol, adj_sllim))
            tp_price = float(self.ex.ccxt.price_to_precision(symbol, tp_price + step))

        return stop_price, stop_limit_price, tp_price

    def _apply_exit_rules_by_pct(self, pos: Position, last_price: float) -> bool:
        """Применяет процентные правила выхода для одной позиции:
        - стоп-лосс при -25% от цены входа;
        - внутри дня тейк-профит при +7%;
        - в конце дня фиксация любой положительной доходности.

        Возвращает True, если позиция была закрыта и дальнейшая обработка символа не нужна.
        """
        if pos is None or pos.qty <= 0 or pos.entry <= 0:
            return False

        pnl_pct = (last_price - pos.entry) / pos.entry

        # 1) Стоп-лосс при -25% от цены входа: фиксируем убыток и увеличиваем счётчик убыточных сделок.
        if pnl_pct <= -0.25:
            self._cancel_position(pos, reason=f"stop_loss_-25pct pnl={pnl_pct:.4f}", exit_price=last_price)
            try:
                self.losses_in_row += 1
            except Exception:
                pass
            return True

        # Текущее локальное время сервера (для правила конца дня)
        now_local = datetime.now().time()

        # 2) Внутридневной тейк-профит при +7% от цены входа: фиксация прибыли, сброс счётчика убыточных сделок.
        if pnl_pct >= 0.07:
            self._cancel_position(pos, reason=f"take_profit_intraday_+7pct pnl={pnl_pct:.4f}", exit_price=last_price)
            try:
                self.losses_in_row = 0
            except Exception:
                pass
            return True

        # 3) В конце дня фиксируем любую положительную доходность.
        # Здесь считаем «концом дня» период после 23:00 локального времени.
        if pnl_pct > 0 and now_local.hour >= 23:
            self._cancel_position(pos, reason=f"take_profit_eod_positive pnl={pnl_pct:.4f}", exit_price=last_price)
            try:
                self.losses_in_row = 0
            except Exception:
                pass
            return True

        return False

    # На каждом тике сверяет желаемые уровни SL/TP с тем, что фактически
    # выставлено на бирже. Если OCO/стопы устарели или отсутствуют,
    # они переустанавливаются с учётом текущего ATR и размера позиции.
    def _reconcile_protective_orders(self, symbol: str, last_close: float, atr_val: float) -> None:
        """Синхронизация логических уровней SL/TP со стопами/тейками на бирже.

        В начале каждого тика проверяет открытые защитные ордера (SL/TP) и при необходимости
        обновляет их: удаляет лишние, двигает уровни или создаёт заново."""
        try:
            open_orders = self.ex.fetch_open_orders(symbol)
        except Exception:
            open_orders = []

        pos = self.positions.get(symbol)

        # If no tracked position exists, cancel stray sell-protective orders to avoid dangling risk.
        if pos is None:
            for o in open_orders:
                try:
                    if str(o.get('side')).lower() == 'sell':
                        oid = o.get('id') or o.get('orderId') or (o.get('info') or {}).get('orderId')
                        if oid:
                            self.ex.cancel_order(str(oid), symbol)
                            log(f"{symbol}: отменён лишний ордер без позиции id={oid}")
                except Exception:
                    continue
            return

        # Compute desired prices from current context using trailing logical stop (pos.stop)
        desired_sl, desired_sllim, desired_tp = self._compute_protective_prices(symbol, pos.entry, pos.stop, last_close)

        # If OCO is unavailable on this account, avoid placing a separate TP order
        # simultaneously with SL, because spot will reserve the full base amount
        # for each sell order leading to insufficient balance. We'll manage TP in
        # logic only and keep a hard SL on exchange.
        if not self.ex.has_oco:
            desired_tp = None

        # Identify current SL/TP orders
        sl_orders, tp_orders = [], []
        for o in open_orders:
            try:
                if str(o.get('side')).lower() != 'sell':
                    continue
                info = o.get('info') or {}
                stop_p = info.get('stopPrice')
                price = o.get('price')
                # Heuristics: stop-limit has stopPrice; TP is plain limit above price
                if stop_p is not None:
                    sl_orders.append(o)
                else:
                    try:
                        if price is not None and float(price) > float(last_close):
                            tp_orders.append(o)
                    except Exception:
                        pass
            except Exception:
                continue

        step = max(self.ex.price_step(symbol), 0.0) or (max(last_close, 1e-12) * 0.001)

        # Determine if changes are needed
        def need_update_sl(o) -> bool:
            try:
                info = o.get('info') or {}
                cur_stop = float(info.get('stopPrice'))
                cur_limit = float(o.get('price')) if o.get('price') is not None else cur_stop - step
                return (abs(cur_stop - desired_sl) > step/2) or (abs(cur_limit - desired_sllim) > step/2) or (cur_stop >= last_close) or (cur_limit >= cur_stop)
            except Exception:
                return True

        def need_update_tp(o) -> bool:
            try:
                cur_price = float(o.get('price'))
                return (desired_tp is None) or (abs(cur_price - desired_tp) > step/2) or (cur_price <= last_close)
            except Exception:
                return True

        sl_needs_change = any(need_update_sl(o) for o in sl_orders) if sl_orders else True  # True if missing
        tp_needs_change = False
        if self.cfg.USE_TP:
            tp_needs_change = (any(need_update_tp(o) for o in tp_orders) if tp_orders else (desired_tp is not None))

        # If quantities on exchange differ from tracked, prefer tracked (rounded) qty
        qty = self.ex.round_qty(symbol, float(pos.qty))
        if qty <= 0:
            return

        # Check minNotional before trying to place
        est_cost = qty * max(1e-12, float(last_close))
        min_cost = self.ex.min_order_cost_quote(symbol, fallback_price=float(last_close))
        if (min_cost is not None) and (est_cost < float(min_cost)):
            # Too small to maintain protective orders — cancel existing
            for o in sl_orders + tp_orders:
                try:
                    oid = o.get('id') or o.get('orderId') or (o.get('info') or {}).get('orderId')
                    if oid:
                        self.ex.cancel_order(str(oid), symbol)
                except Exception:
                    pass
            log(f"{symbol}: отменены защитные ордера — позиция ниже minNotional")
            return

        # Apply updates
        try:
            if self.cfg.USE_TP and self.ex.has_oco and (sl_needs_change or tp_needs_change):
                # cancel all existing sell orders and recreate OCO fresh
                for o in sl_orders + tp_orders:
                    try:
                        oid = o.get('id') or o.get('orderId') or (o.get('info') or {}).get('orderId')
                        if oid:
                            self.ex.cancel_order(str(oid), symbol)
                    except Exception:
                        pass
                self.ex.create_oco_sell(symbol, qty, desired_tp, desired_sl, desired_sllim)
                log(f"{symbol}: OCO обновлён sl={fmt_float(desired_sl,8)} slLim={fmt_float(desired_sllim,8)} tp={fmt_float(desired_tp if desired_tp else 0,8)}")
                return

            # Otherwise, handle independently
            if sl_needs_change:
                for o in sl_orders:
                    try:
                        oid = o.get('id') or o.get('orderId') or (o.get('info') or {}).get('orderId')
                        if oid:
                            self.ex.cancel_order(str(oid), symbol)
                    except Exception:
                        pass
                self.ex.create_stop_loss_limit(symbol, qty, desired_sl, desired_sllim)
                log(f"{symbol}: SL обновлён sl={fmt_float(desired_sl,8)} slLim={fmt_float(desired_sllim,8)}")

            # TP order only if OCO is supported
            if self.cfg.USE_TP and self.ex.has_oco:
                if tp_needs_change:
                    for o in tp_orders:
                        try:
                            oid = o.get('id') or o.get('orderId') or (o.get('info') or {}).get('orderId')
                            if oid:
                                self.ex.cancel_order(str(oid), symbol)
                        except Exception:
                            pass
                    if desired_tp is not None:
                        self.ex.create_limit_sell(symbol, qty, desired_tp)
                        log(f"{symbol}: TP обновлён tp={fmt_float(desired_tp,8)}")
            else:
                if self.cfg.USE_TP and not self.ex.has_oco:
                    log(f"{symbol}: OCO недоступен — ставим только SL без отдельного TP, чтобы не резервировать двойной объём", True)
        except Exception as e:
            log(f"{symbol}: reconcile protective orders failed: {e}")
    # Bootstrap: поиск уже имеющихся спот-позиций по указанным символам.
    # Если на кошельке есть свободный остаток базовой валюты, бот создаёт
    # для него Position и (в live-режиме) пытается автоматически поставить
    # защитные ордера. Это позволяет подключить бота к уже открытым ручным
    # сделкам.
    def bootstrap_existing_positions(self):
        """Сканирует свободные остатки базовых валют для всех символов и, если они есть, добавляет их как активные позиции
        с разумными стоп/тейк уровнями, чтобы стратегия могла управлять и закрывать их."""
        for symbol in self.cfg.MARKETS:
            try:
                # Учитываем не только free, но и зарезервированный (used) объём,
                # чтобы подхватывать позиции, которые целиком висят в ордерах.
                bal = self.ex.ccxt.fetch_balance()
                base = symbol.split('/')[0]
                free = float((bal.get('free') or {}).get(base, 0.0) or 0.0)
                used = float((bal.get('used') or {}).get(base, 0.0) or 0.0)
                total = float((bal.get('total') or {}).get(base, free + used) or (free + used))
                base_qty = total if total > 0 else (free + used)
                base_qty_free = free
            except Exception as e:
                log(f"{symbol}: bootstrap — не удалось получить баланс базовой валюты: {e}")
                continue
            if base_qty <= 0:
                continue

            # Проверка minNotional на случай пыли
            last = self.ex.last_price(symbol)
            min_cost = self.ex.min_order_cost_quote(symbol, fallback_price=last)
            est_cost = base_qty * last
            if (min_cost is not None) and (est_cost < float(min_cost)):
                log(f"{symbol}: bootstrap — остаток слишком мал для ордера (≈{est_cost:.2f} < min≈{float(min_cost):.2f})")
                continue

            # Данные для ATR/TP/SL
            try:
                tf_df = self.ex.fetch_ohlcv(symbol, self.cfg.TIMEFRAME, max(60, self.cfg.LOOKBACK))
            except Exception as e:
                log(f"{symbol}: bootstrap — не удалось загрузить OHLCV: {e}")
                continue
            df = tf_df.copy()
            df['atr'] = atr(df, 14)
            last_row = df.iloc[-1]
            atr_val = float(last_row['atr'])
            entry = self.ex.avg_buy_price(symbol) or last

            # Начальные уровни (как в _place_orders)
            stop = entry - self.cfg.ATR_K * atr_val
            tp = None
            if self.cfg.USE_TP:
                tp = entry + self.cfg.TP_R_MULT * (entry - stop)

            # Пытаемся автоматически поставить защитные ордера для ручных позиций (только в live)
            qty_place = self.ex.round_qty(symbol, float(base_qty_free))
            if self.cfg.MODE == 'live':
                if qty_place <= 0:
                    log(f"{symbol}: bootstrap — пропускаем постановку защитных ордеров, свободный остаток 0 (free={base_qty_free:.8f}, used={used:.8f})", True)
                else:
                    if qty_place < base_qty:
                        log(f"{symbol}: bootstrap — ставим ордера только на свободный остаток free={qty_place:.8f} из total≈{base_qty:.8f}", True)
                    try:
                        last_px = self.ex.last_price(symbol)
                        est_cost = qty_place * last_px
                        min_cost = self.ex.min_order_cost_quote(symbol, fallback_price=last_px)
                        if (min_cost is None) or (est_cost >= float(min_cost)):
                            stop_price = float(stop)
                            step = max(self.ex.price_step(symbol), 0.0) or (last_px * 0.001)
                            minp = self.ex.min_price(symbol)
                            # Ensure correct relationships for SELL OCO: tp > last, stop < last, stop_limit < stop
                            stop_price = min(stop_price, last_px - step)
                            stop_limit_price = max(minp, stop_price - step)
                            raw_tp = float(tp) if tp else None
                            tp_price = None
                            if self.cfg.USE_TP and raw_tp is not None:
                                tp_price = max(raw_tp, last_px + step, stop_price + step)
                            # Round all to precision
                            stop_price = float(self.ex.ccxt.price_to_precision(symbol, stop_price))
                            stop_limit_price = float(self.ex.ccxt.price_to_precision(symbol, stop_limit_price))
                            if tp_price is not None:
                                tp_price = float(self.ex.ccxt.price_to_precision(symbol, tp_price))
                            
                            # Final sanity: enforce strict inequalities after rounding, without going below minimal price
                            if tp_price is not None and not (tp_price > last_px and stop_price < last_px and stop_limit_price < stop_price):
                                adj_stop = max(minp, stop_price - step)
                                adj_sllim = max(minp, stop_limit_price - step)
                                stop_price = float(self.ex.ccxt.price_to_precision(symbol, adj_stop))
                                stop_limit_price = float(self.ex.ccxt.price_to_precision(symbol, adj_sllim))
                                tp_price = float(self.ex.ccxt.price_to_precision(symbol, tp_price + step))
                            if self.cfg.USE_TP and tp_price is not None and self.ex.has_oco:
                                self.ex.create_oco_sell(symbol, qty_place, tp_price, stop_price, stop_limit_price)
                            else:
                                # OCO недоступен — ставим только SL. Отдельный TP не выставляем,
                                # чтобы не резервировать двойной объём и не получать ошибку баланса.
                                self.ex.create_stop_loss_limit(symbol, qty_place, stop_price, stop_limit_price)
                                if self.cfg.USE_TP and tp_price is not None and not self.ex.has_oco:
                                    log(f"{symbol}: OCO недоступен — TP не выставлен (только SL)")
                            log(f"{symbol}: bootstrap — поставлены защитные ордера для ручной позиции qty={qty_place:.8f} sl≈{fmt_float(stop,8)} tp≈{fmt_float(tp if tp else 0,8) if self.cfg.USE_TP else 'None'}")
                        else:
                            log(f"{symbol}: bootstrap — ручная позиция слишком мала для постановки ордеров (≈{est_cost:.2f} < min≈{float(min_cost):.2f})")
                    except Exception as e:
                        log(f"{symbol}: bootstrap — не удалось поставить защитные ордера для ручной позиции: {e}")

            # Округление для логов
            entry_str = fmt_float(entry, 8)
            stop_str = fmt_float(stop, 8)
            tp_str = fmt_float(tp, 8) if tp is not None else 'None'

            tracked_qty = qty_place if qty_place > 0 else self.ex.round_qty(symbol, float(base_qty))
            self.positions[symbol] = Position(symbol, 'long', float(tracked_qty), float(entry), float(stop), float(tp) if tp else None)
            log(f"{symbol}: bootstrap — обнаружен баланс {base_qty:.8f} {symbol.split('/')[0]}, позиция создана entry={entry_str} stop={stop_str} tp={tp_str}")
    """
    Стратегия (лонг-только, учебный пример):
      Вход (на младшем ТФ, обычно 1m):
        - Последняя свеча закрылась выше SMA20.
        - Пробой максимума предыдущей свечи (last.close > prev.high).
        - ATR% в диапазоне [ATR_PCT_MIN, ATR_PCT_MAX].
        - При включённом FAST_MODE дополнительно проверяются:
          * RSI на старшем ТФ (FAST_HTF) — должен быть не ниже FAST_RSI_MIN.
          * Объём текущей свечи на младшем ТФ — не ниже SMA(объёма, FAST_MIN_VOL_SMA).

      Размер позиции:
        - В текущей реализации размер позиции рассчитывается не от RISK_PCT,
          а от фиксированной целевой стоимости входа в котируемой валюте
          (по умолчанию ≈10 EUR на сделку, см. _position_size).
        - RISK_PCT оставлен в конфиге на будущее как возможный альтернативный
          режим расчёта объёма (риск от equity).

      Стоп-лосс и тейк-профит:
        - При открытии позиции считается логический (виртуальный) стоп:
            * либо entry - ATR_K * ATR,
            * либо entry - FIXED_STOP_EUR, если FIXED_STOP_EUR > 0
              (интерпретируется как смещение цены, а не как риск в EUR).
        - На основе расстояния entry - stop вычисляется TP:
            tp = entry + TP_R_MULT * (entry - stop).
        - Внутренние уровни стопа и TP сохраняются в объекте Position и
          используются для дальнейшего управления позицией.

      Защитные ордера на бирже:
        - Фактические SL/TP ордера на бирже выставляются и обновляются не
          в момент входа, а в отдельном шаге _reconcile_protective_orders,
          который вызывается на каждом тике.
        - Если аккаунт поддерживает OCO, ставится связка TP+SL (OCO-ордер).
          В противном случае бот выставляет только стоп-лимит (SL), а TP
          остаётся логическим и обрабатывается в коде.
        - Все уровни проходятся через фильтры биржи (PRICE_FILTER,
          PERCENT_PRICE_BY_SIDE, minNotional), чтобы избежать ошибок InvalidOrder.

      Дневные ограничения и риск-менеджмент:
        - Отслеживается дневная equity в котируемой валюте (по умолчанию EUR).
        - MAX_DAILY_DD_PCT ограничивает максимально допустимую дневную
          просадку; при её превышении новые входы блокируются до следующего дня.
        - Параметр MAX_LOSSES_IN_ROW используется в on_tick для остановки
          входов после серии убыточных сделок.

      ВАЖНО: код приведён в образовательных целях. Перед реальной торговлей
      обязательно тестируйте бота в paper-режиме, проверяйте логи, ордера,
      точность расчётов и учитывайте комиссии, спред и скольжение.
    """
    def __init__(self, cfg: Config, ex: Exchange):
        # Сохраняем конфиг и обёртку биржи, инициализируем состояние стратегии:
        # открытые позиции, дневную точку отсчёта по equity и счётчик
        # подряд идущих убыточных сделок.
        self.cfg = cfg
        self.ex = ex
        self.positions: Dict[str, Position] = {}
        self.realized_pnl_eur: float = 0.0
        self.daily_start_equity_eur: Optional[float] = None
        self.losses_in_row: int = 0
        self.current_date: date = date.today()

    # Сброс дневных ограничений при смене календарного дня:
    # точка отсчёта по equity и счётчик подряд идущих убыточных сделок.
    def _reset_daily_limits_if_new_day(self):
        today = date.today()
        if today != self.current_date:
            self.current_date = today
            self.daily_start_equity_eur = None
            self.losses_in_row = 0

    # Гарантирует, что дневная базовая величина equity рассчитана.
    # Используется для контроля дневной просадки (MAX_DAILY_DD_PCT).
    def _ensure_daily_equity_baseline(self, quote='EUR'):
        if self.daily_start_equity_eur is None:
            try:
                self.daily_start_equity_eur = self.ex.balance_total_in_quote(quote)
            except Exception:
                # If we cannot fetch balance (paper mode), fall back to last known equity
                self.daily_start_equity_eur = 0.0

    # Проверяет, не превышен ли дневной лимит просадки.
    # Если просадка больше MAX_DAILY_DD_PCT, новые входы блокируются
    # до следующего календарного дня.
    def _check_daily_dd_limit(self, quote='EUR') -> bool:
        self._ensure_daily_equity_baseline(quote)
        try:
            current = self.ex.balance_total_in_quote(quote)
        except Exception:
            # Assume not breached if cannot fetch
            return True
        if self.daily_start_equity_eur == 0:
            return True
        dd = (current - self.daily_start_equity_eur) / self.daily_start_equity_eur
        return dd >= -self.cfg.MAX_DAILY_DD_PCT * 1.001  # small buffer

    # Генерация сигнала на вход в лонг.
    # На младшем ТФ (TIMEFRAME, обычно 1m):
    #  - последний close выше SMA20
    #  - пробой максимума предыдущей свечи (last.close > prev.high)
    #  - ATR% в пределах [ATR_PCT_MIN, ATR_PCT_MAX]
    # При включённом FAST_MODE дополнительно:
    #  - RSI на старшем ТФ (FAST_HTF) >= FAST_RSI_MIN
    #  - текущий объём >= SMA объёма за FAST_MIN_VOL_SMA свечей.
    # Возвращает флаг will_long и контекст ctx для логирования.
    def _signal(self, symbol: str, tf_df: pd.DataFrame, htf_df: Optional[pd.DataFrame]) -> Tuple[bool, Dict]:
        # Считаем индикаторы на 1m: SMA20, ATR (Wilder), ATR% и средний объём.
        df = tf_df.copy()
        df['sma20'] = sma(df['close'], 20)
        df['atr'] = atr(df, 14)
        df['atr_pct'] = df['atr'] / df['close']
        df['vol_sma'] = sma(df['volume'], max(2, self.cfg.FAST_MIN_VOL_SMA))

        # Базовый сигнал пробоя: последняя свеча закрылась выше SMA20 и выше максимума предыдущей свечи.
        prev = df.iloc[-2]
        last = df.iloc[-1]
        cond_breakout = (last['close'] > last['sma20']) and (last['close'] > prev['high'])

        # Фильтр по волатильности: слишком низкая/высокая волатильность отбрасывает вход.
        atr_ok = (last['atr_pct'] >= self.cfg.ATR_PCT_MIN) and (last['atr_pct'] <= self.cfg.ATR_PCT_MAX)

        # Дополнительные фильтры FAST (HTF RSI и объём на 1m относительно своей SMA).
        if self.cfg.FAST_MODE and htf_df is not None:
            hdf = htf_df.copy()
            hdf['rsi'] = rsi(hdf['close'], 14)
            rsi_ok = hdf['rsi'].iloc[-1] >= self.cfg.FAST_RSI_MIN
            vol_ok = last['volume'] >= (df['vol_sma'].iloc[-1] if not np.isnan(df['vol_sma'].iloc[-1]) else 0)
        else:
            rsi_ok = True
            vol_ok = True

        # Итоговый флаг входа: все условия должны выполниться.
        will_long = bool(cond_breakout and atr_ok and rsi_ok and vol_ok)
        ctx = {
            'last_close': float(last['close']),
            'prev_high': float(prev['high']),
            'atr': float(last['atr']),
            'atr_pct': float(last['atr_pct']),
            'rsi_ok': bool(rsi_ok),
            'vol_ok': bool(vol_ok),
            'cond_breakout': bool(cond_breakout),
            'atr_ok': bool(atr_ok),
        }
        return will_long, ctx

    # Расчёт размера позиции от фиксированной суммы входа в котируемой валюте.
    # В текущей конфигурации бот всегда старается открыть сделку примерно на 10 EUR,
    # независимо от ширины стопа/волатильности. ATR здесь может использоваться для расчёта
    # виртуального стопа и TP, но не влияет на объём.
    def _position_size(self, symbol: str, entry: float, atr_val: float, quote='EUR') -> float:
        # ФИКСИРОВАННЫЙ РАЗМЕР СДЕЛКИ: целевая стоимость позиции ≈ 10 EUR.
        target_cost = 10.0  # EUR

        if entry <= 0:
            return 0.0

        # Базовый объём по целевой сумме сделки (без учёта биржевых лимитов)
        qty_raw = target_cost / float(entry)

        # Ограничение сверху по доступному свободному балансу котируемой валюты
        qty_cap_balance = self.ex.max_buy_qty(symbol, 0.995)
        qty = min(qty_raw, qty_cap_balance)

        # Early check against explicit min amount to avoid precision errors
        try:
            info = self.ex.market_info(symbol)
            limits_amount = (info.get('limits') or {}).get('amount') or {}
            min_amount = limits_amount.get('min')
            if min_amount is not None:
                min_amount = float(min_amount)
                if qty < min_amount:
                    log(f"{symbol}: proposed qty {qty:.8f} < min_amount {min_amount:.8f} — skip position sizing", True)
                    return 0.0
        except Exception:
            pass

        qty = self.ex.round_qty(symbol, qty)

        # If rounding/limits reduce qty below the allowed minimum, skip this trade
        if qty <= 0:
            return 0.0

        # Проверка минимальной стоимости ордера (minNotional)
        last = max(1e-12, float(entry))
        est_cost = qty * last
        min_cost = self.ex.min_order_cost_quote(symbol, fallback_price=last)
        if (min_cost is not None) and (est_cost < float(min_cost)):
            return 0.0
        return qty

    def _place_orders(self, symbol: str, qty: float, entry: float, atr_val: float) -> Optional[Position]:
        # Выставление ордеров для нового входа:
        #  - в paper-режиме — создаётся только виртуальная Position с логическим стопом и TP;
        #  - в live — отправляется рыночный ордер на покупку (market buy), без мгновенной
        #    постановки защитных ордеров.
        #
        # На этом шаге стоп-лосс и тейк-профит рассчитываются как логические уровни и сохраняются
        # в объекте позиции. Фактические биржевые защитные ордера (SL/TP или OCO) выставляются
        # и обновляются отдельно в _reconcile_protective_orders, чтобы учитывать требования
        # биржи по ценовым фильтрам, minNotional и возможные изменения trailing-стопа.
        stop_virtual = entry - (self.cfg.FIXED_STOP_EUR if self.cfg.FIXED_STOP_EUR > 0 else self.cfg.ATR_K * atr_val)
        tp = None
        if self.cfg.USE_TP:
            tp = entry + self.cfg.TP_R_MULT * (entry - stop_virtual)

        # Подробный комментарий в лог: как именно рассчитаны виртуальный стоп и TP.
        # Это помогает понять, почему TP может выглядеть «далёким» от входа,
        # особенно когда FIXED_STOP_EUR интерпретируется как смещение цены.
        if self.cfg.VERBOSE:
            try:
                risk_per_unit = entry - stop_virtual
                log(
                    f"{symbol}: расчёт уровней для входа — entry={fmt_float(entry, 8)}, "
                    f"virtual_stop={fmt_float(stop_virtual, 8)}, расстояние_R={fmt_float(risk_per_unit, 8)}, "
                    f"TP_R_MULT={self.cfg.TP_R_MULT:.3f}, tp={fmt_float(tp if tp is not None else 0.0, 8)} "
                    f"(FIXED_STOP_EUR интерпретируется как смещение цены, а не как риск в EUR)",
                    True
                )
            except Exception:
                # В случае любых проблем с форматированием не мешаем основной логике входа.
                pass

        if self.cfg.MODE == 'paper':
            # В paper-режиме храним виртуальный стоп для расчётов, но логически
            # позиция закрывается только по TP или вручную.
            return Position(symbol, 'long', qty, entry, stop_virtual, tp)

        # LIVE
        # Safety checks: free balance and minNotional before отправки ордера
        quote_ccy = symbol.split('/')[1] if '/' in symbol else 'EUR'
        free_quote = self.ex.quote_free(quote_ccy)
        est_cost = float(qty) * float(entry)
        min_cost = self.ex.min_order_cost_quote(symbol, fallback_price=float(entry))
        log(f"{symbol}: try market buy qty={qty:.8f}, price≈{entry:.8g}, cost≈{est_cost:.2f} {quote_ccy}, free≈{free_quote:.2f} {quote_ccy}, min_cost≈{0 if min_cost is None else float(min_cost):.2f}", True)
        if qty <= 0 or (min_cost is not None and est_cost < float(min_cost)) or est_cost > free_quote * 0.999:
            log(f"{symbol}: skip buy — qty/cost not feasible (min_cost/free check)", True)
            return None

        # 1) Рыночная покупка
        buy = self.ex.create_market_buy(symbol, qty)
        filled_price = float(buy.get('average', buy.get('price', entry)) or entry)

        # После исполнения market buy реальный доступный объём базовой валюты
        # может быть чуть меньше из-за комиссий. Поэтому для постановки TP и
        # учёта позиции используем фактический free-баланс, а не исходный qty.
        try:
            base_free_after = self.ex.base_free(symbol)
            qty_for_tp = self.ex.round_qty(symbol, base_free_after)
        except Exception:
            qty_for_tp = self.ex.round_qty(symbol, qty)

        if qty_for_tp <= 0:
            log(f"{symbol}: qty_for_tp <= 0 после market buy — позиция не будет открыта", True)
            return None

        # 2) Пересчитываем виртуальный стоп и TP от фактической цены входа
        stop_virtual = filled_price - (self.cfg.FIXED_STOP_EUR if self.cfg.FIXED_STOP_EUR > 0 else self.cfg.ATR_K * atr_val)
        tp_price = None
        if self.cfg.USE_TP:
            base_tp = filled_price + self.cfg.TP_R_MULT * (filled_price - stop_virtual)
            step = max(self.ex.price_step(symbol), 0.0) or (filled_price * 0.001)
            tp_raw = max(base_tp, filled_price + step)
            tp_price = float(self.ex.ccxt.price_to_precision(symbol, tp_raw))

            try:
                # Ставим только лимитный TP, без стоп-лосса.
                self.ex.create_limit_sell(symbol, qty_for_tp, tp_price)
            except Exception as e:
                log_error(f"{symbol}: could not place TP limit order", e)

        return Position(symbol, 'long', qty_for_tp, filled_price, stop_virtual, tp_price if tp_price is not None else None)

    # Закрытие позиции в контексте стратегии: удаление из трекера и запись
    # события в лог. Фактическое закрытие на бирже происходит за счёт
    # исполнения защитных ордеров. В live-режиме дополнительно пытаемся
    # продать весь доступный остаток базовой валюты рыночным ордером.
    def _cancel_position(self, pos: Position, reason: str, exit_price: Optional[float] = None):
        def _safe_float(*vals) -> Optional[float]:
            for v in vals:
                try:
                    if v is None:
                        continue
                    return float(v)
                except Exception:
                    continue
            return None

        quote_ccy = pos.symbol.split('/')[1] if '/' in pos.symbol else 'QUOTE'

        # Оценка прибыли/убытка в котируемой валюте (обычно EUR)
        est_exit_px = exit_price
        if est_exit_px is None:
            try:
                est_exit_px = self.ex.last_price(pos.symbol)
            except Exception:
                est_exit_px = None
        if est_exit_px is None or est_exit_px <= 0:
            est_exit_px = pos.entry
        est_pnl_quote = (est_exit_px - pos.entry) * pos.qty

        log_trade(
            f"EXIT {pos.symbol} side=long reason={reason} exit_px={fmt_float(est_exit_px,8)} pnl_quote={est_pnl_quote:.4f} {quote_ccy}"
        )

        if self.cfg.MODE == 'live':
            # 1) Пытаемся отменить все оставшиеся ордера по символу (если что-то висит).
            try:
                self.ex.cancel_all_orders(pos.symbol)
            except Exception as e:
                log_error(f"{pos.symbol}: failed to cancel open orders on exit", e)

            # 2) Берём весь свободный остаток базовой валюты и продаём его по рынку.
            try:
                qty_free = self.ex.base_free(pos.symbol)
                qty_free = self.ex.round_qty(pos.symbol, qty_free)
                if qty_free > 0:
                    order = self.ex.create_market_sell(pos.symbol, qty_free)
                    info = order if isinstance(order, dict) else {}
                    filled_qty = _safe_float(info.get('filled'), info.get('amount'), info.get('info', {}).get('executedQty')) or qty_free
                    fill_price = _safe_float(
                        info.get('average'),
                        info.get('price'),
                        info.get('cost', 0) / filled_qty if filled_qty else None,
                        info.get('info', {}).get('avgPrice'),
                        info.get('info', {}).get('price')
                    )
                    if fill_price is None:
                        fill_price = est_exit_px
                    actual_pnl_quote = (fill_price - pos.entry) * filled_qty
                    log_trade(
                        f"FORCE_EXIT {pos.symbol} market_sell_all qty={qty_free:.8f} fill_px={fmt_float(fill_price,8)} pnl_quote={actual_pnl_quote:.4f} {quote_ccy}"
                    )
                else:
                    log(f"{pos.symbol}: no free base balance to force-sell on exit", True)
            except Exception as e:
                log_error(f"{pos.symbol}: failed to force-sell on exit", e)

        # 3) В любом случае удаляем позицию из внутреннего трекера.
        self.positions.pop(pos.symbol, None)

    # Основной цикл стратегии (один тик).
    # 1) Проверяет дневные лимиты по просадке
    # 2) Для каждого символа загружает данные, считает индикаторы и сигнал
    # 3) Синхронизирует и обновляет защитные ордера
    # 4) Обновляет уже открытые позиции (стоп/тейк, трейлинг)
    # 5) При наличии сигнала и отсутствии ограничений пытается открыть новую позицию.
    def on_tick(self):
        # Один цикл работы: проверка дневных лимитов, загрузка данных, сигналы,
        # управление открытыми позициями и новые входы.
        self._reset_daily_limits_if_new_day()
        if not self._check_daily_dd_limit('EUR'):
            log("Daily drawdown limit reached — pausing new entries.")
            return

        # Проходимся по всем указанным в .env торговым парам.
        for symbol in self.cfg.MARKETS:
            try:
                tf_df = self.ex.fetch_ohlcv(symbol, self.cfg.TIMEFRAME, self.cfg.LOOKBACK)
            except Exception as e:
                log(f"{symbol}: fetch {self.cfg.TIMEFRAME} failed: {e}", self.cfg.VERBOSE)
                continue

            htf_df = None
            if self.cfg.FAST_MODE:
                try:
                    htf_df = self.ex.fetch_ohlcv(symbol, self.cfg.FAST_HTF, max(60, int(self.cfg.LOOKBACK/5)))
                except Exception as e:
                    log(f"{symbol}: fetch {self.cfg.FAST_HTF} failed: {e}", self.cfg.VERBOSE)

            # Entry check
            will_long, ctx = self._signal(symbol, tf_df, htf_df)
            last_close = ctx['last_close']
            atr_val = ctx['atr']

            # Ранее здесь происходила синхронизация защитных ордеров (SL/TP) на бирже.
            # В текущей конфигурации без стоп-лосса эту логику отключаем: TP
            # выставляется один раз при входе в позицию, а дополнительных стоп-ордеров
            # стратегия не создаёт.
            pass

            # Менеджмент открытой позиции: применяем процентные правила выхода
            # (SL -25%, внутридневной TP +7%, фиксация плюса в конце дня).
            if symbol in self.positions:
                pos = self.positions[symbol]

                # Берём актуальную рыночную цену для оценки PnL; при ошибке падаем обратно к last_close.
                try:
                    live_px = self.ex.last_price(symbol)
                except Exception:
                    live_px = last_close

                if self._apply_exit_rules_by_pct(pos, live_px):
                    # Позиция уже закрыта одним из правил, новые входы по символу
                    # в этом тике не рассматриваем.
                    continue

                # Пока позиция открыта, новые входы по этому символу не допускаются.
                continue

            # Риск-контроль по серии убыточных сделок.
            if self.losses_in_row >= self.cfg.MAX_LOSSES_IN_ROW:
                log(f"{symbol}: skip (loss streak {self.losses_in_row} >= {self.cfg.MAX_LOSSES_IN_ROW})", self.cfg.VERBOSE)
                continue

            if will_long:
                # Доп. защита: если уже есть позиция или открытые ордера — пропускаем новый вход
                if self._has_position_or_pending(symbol):
                    if self.cfg.VERBOSE:
                        log(f"{symbol}: skip — already have position or pending orders")
                    continue
                qty = self._position_size(symbol, last_close, atr_val, 'EUR')
                if qty <= 0:
                    log(f"{symbol}: qty too small", self.cfg.VERBOSE)
                    continue
                pos = self._place_orders(symbol, qty, last_close, atr_val)
                if pos:
                    self.positions[symbol] = pos
                    entry_str = fmt_float(pos.entry, 8)
                    stop_str = fmt_float(pos.stop, 8)
                    tp_str = fmt_float(pos.tp, 8) if pos.tp is not None else "None"
                    log_trade(f"ENTER {symbol} side=long qty={pos.qty:.8f} entry={entry_str} stop={stop_str} tp={tp_str}")
                else:
                    log(f"{symbol}: order placement failed", True)
            else:
                if self.cfg.VERBOSE:
                    log(f"{symbol}: no entry — ctx=" + format_ctx_str(ctx, 8))

# =========================
# Главный цикл
# =========================
# Точка входа в программу.
# 1) Загружает конфигурацию из .env
# 2) Настраивает логирование
# 3) Создаёт обёртку биржи и стратегию
# 4) Выполняет префлайт-проверку баланса и фильтрацию рынков
# 5) Выполняет bootstrap уже имеющихся позиций
# 6) Запускает бесконечный цикл, в котором периодически вызывается on_tick().
def main():
    # Читаем конфиг из .env и подготавливаем все параметры стратегии.
    # Точка входа: читаем конфиг, проверяем наличие ключей для live, создаём биржу и стратегию.
    cfg = load_config()
    # Инициализация структурированных логов: app.log, trades.log, errors.log
    setup_logging('logs')
    log(f"Starting bot — mode={cfg.MODE} markets={cfg.MARKETS} timeframe={cfg.TIMEFRAME}", True)

    if cfg.MODE == 'live' and not cfg.API_KEY:
        log('LIVE mode requires API_KEY/API_SECRET set in .env — aborting.', True)
        return

    # Создаём обёртку над ccxt и загружаем информацию о рынках.
    ex = Exchange(cfg)
    # Предзапуск: баланс в котируемой валюте (EUR) и проверка достаточности средств
    # Префлайт: оцениваем общий баланс в EUR и проверяем, по каким рынкам
    # депозит позволяет выставлять минимальные ордера. Остальные рынки
    # сразу исключаются из анализа.
    try:
        eur_balance = ex.balance_total_in_quote('EUR')
        log(f"Оценка эквивалента баланса ≈ {eur_balance:.2f} EUR (по рын. котировкам)", True)
        for sym in cfg.MARKETS:
            ok, need = ex.affordable(sym, eur_balance)
            if need is None:
                log(f"{sym}: нельзя оценить минимальную стоимость ордера (нет данных о лимитах)")
            else:
                status = 'ДОСТАТОЧНО' if ok else 'НЕДОСТАТОЧНО'
                log(f"{sym}: мин. ордер ≈ {need:.2f} EUR — {status}")
        # Отфильтруем список рынков по достаточности средств и будем анализировать только их
        tradable_markets = filter_markets_by_affordability(ex, cfg.MARKETS, eur_balance, verbose=True)
        if not tradable_markets:
            log("Нет рынков, удовлетворяющих минимальному требованию по стоимости ордера — анализ будет пропущен.")
        else:
            log(f"Будут анализироваться только рынки: {tradable_markets}")
        cfg.MARKETS = tradable_markets
    except Exception as e:
        log_error("Не удалось получить баланс/лимиты", e)

    # Создаём экземпляр стратегии, которая будет обрабатывать тики.
    strat = BreakoutWithATRAndRSI(cfg, ex)

    # Подхватываем уже существующие спот-позиции по указанным рынкам
    # и, при возможности, автоматически выставляем для них защитные ордера.
    # Добавляем в управление уже имеющиеся активы на споте
    try:
        strat.bootstrap_existing_positions()
    except Exception as e:
        log_error("Bootstrap existing positions failed", e)

    # Главный бесконечный цикл: на каждом шаге вызывается on_tick(),
    # после чего делается пауза SLEEP_SEC секунд.
    # Главный цикл: вызываем on_tick() с паузой SLEEP_SEC.
    while True:
        try:
            strat.on_tick()
        except Exception as e:
            log_error('Error in on_tick', e)
        time.sleep(max(1, cfg.SLEEP_SEC))

if __name__ == '__main__':
    main()
