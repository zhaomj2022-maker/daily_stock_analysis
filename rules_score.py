#!/usr/bin/env python3
"""
规则技术面打分脚本（看板2的第二意见）

方法：确定性规则，不用 LLM。每天收盘后拉 16 只核心股的日线，
用 7 个信号投票打分（每个信号 +1 / 0 / -1），输出一张对照表，
和 daily_stock_analysis 的 LLM 结论并排比较。

数据源：Yahoo Finance chart API（免费，不用 key），失败自动换 query2 重试。

7 个信号：
  1. 趋势：收盘 > EMA50 为 +1，< EMA50 为 -1
  2. 长趋势：收盘 > EMA200 为 +1，< EMA200 为 -1
  3. 短期动量：EMA20 > EMA50 为 +1，< EMA50 为 -1
  4. RSI(14)：<30 超卖为 +1，>70 超买为 -1，其余 0
  5. MACD：柱状图 >0 为 +1，<0 为 -1
  6. 位置：收盘在 20 日最高价 3% 以内为 +1，在 20 日最低价 3% 以内为 -1
  7. 量价：成交量 > 20 日均量 x1.2 且上涨为 +1，且下跌为 -1

分数 -7 ~ +7，对应结论：
  >= +3 偏多；+1 ~ +2 偏多（弱）；0 中性；-1 ~ -2 偏空（弱）；<= -3 偏空

这只是技术面参考，不是投资建议。LLM 结论和规则结论打架时，
先等等、多看一眼，而不是照单动手。

环境变量：
  STOCK_LIST      逗号分隔的股票代码，默认 16 只核心股
  EMAIL_SENDER    发件 Gmail（有则发邮件，无则只打印）
  EMAIL_PASSWORD  Gmail 应用专用密码
  EMAIL_RECEIVERS 收件邮箱（默认发件人自己）
"""

import os
import smtplib
import sys
import time
from datetime import datetime
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import requests

DEFAULT_TICKERS = (
    "VIXY,CBUS,MSFT,NVDA,TSLA,AMD,HAFN,GOOGL,"
    "UAMY,GLD,MMM,OUST,NAT,VST,PLTR,IREN"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def fetch_daily(ticker, retries=3):
    """从 Yahoo chart API 拉一年日线，返回 DataFrame（Close/Volume，日期索引）。"""
    urls = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=1y&interval=1d",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range=1y&interval=1d",
    ]
    last_err = None
    for url in urls:
        for _ in range(retries):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                resp.raise_for_status()
                result = resp.json()["chart"]["result"][0]
                ts = result["timestamp"]
                quote = result["indicators"]["quote"][0]
                adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
                closes = adj if adj else quote["close"]
                dates = pd.to_datetime(ts, unit="s", utc=True).date
                df = pd.DataFrame({"Close": closes, "Volume": quote["volume"]},
                                  index=pd.DatetimeIndex(dates))
                df = df.dropna(subset=["Close"])
                if len(df) >= 210:
                    return df
                last_err = ValueError(f"仅 {len(df)} 根日线")
            except Exception as e:
                last_err = e
                time.sleep(2)
    raise ValueError(f"{ticker}: 拉数失败（{last_err}）")


def rsi_wilder(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def score_ticker(ticker):
    """返回一只股票的打分结果；失败时抛异常由调用方记录。"""
    df = fetch_daily(ticker)
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    rsi = rsi_wilder(close, 14)
    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    change_pct = (last_close - prev_close) / prev_close * 100
    last_date = close.index[-1].strftime("%Y-%m-%d")
    last_rsi = float(rsi.iloc[-1])
    last_hist = float(macd_hist.iloc[-1])
    hi20 = float(close.tail(20).max())
    lo20 = float(close.tail(20).min())
    avg_vol20 = float(volume.tail(20).mean())
    last_vol = float(volume.iloc[-1])

    votes = {}
    votes["趋势EMA50"] = 1 if last_close > float(ema50.iloc[-1]) else -1
    votes["长趋势EMA200"] = 1 if last_close > float(ema200.iloc[-1]) else -1
    votes["短期EMA20/50"] = 1 if float(ema20.iloc[-1]) > float(ema50.iloc[-1]) else -1
    votes["RSI"] = 1 if last_rsi < 30 else (-1 if last_rsi > 70 else 0)
    votes["MACD"] = 1 if last_hist > 0 else -1
    votes["位置"] = 1 if last_close >= hi20 * 0.97 else (-1 if last_close <= lo20 * 1.03 else 0)
    if last_vol > avg_vol20 * 1.2:
        votes["量价"] = 1 if change_pct > 0 else -1
    else:
        votes["量价"] = 0

    score = sum(votes.values())
    if score >= 3:
        verdict = "偏多"
    elif score >= 1:
        verdict = "偏多（弱）"
    elif score == 0:
        verdict = "中性"
    elif score >= -2:
        verdict = "偏空（弱）"
    else:
        verdict = "偏空"

    labels = {
        "趋势EMA50": ("站上EMA50", "跌破EMA50"),
        "长趋势EMA200": ("站上EMA200", "跌破EMA200"),
        "短期EMA20/50": ("EMA20>EMA50", "EMA20<EMA50"),
        "RSI": ("RSI超卖", "RSI超买"),
        "MACD": ("MACD>0", "MACD<0"),
        "位置": ("接近20日高点", "接近20日低点"),
        "量价": ("放量上涨", "放量下跌"),
    }
    signals = []
    for name, v in votes.items():
        if v > 0:
            signals.append(labels[name][0])
        elif v < 0:
            signals.append(labels[name][1])
    return {
        "ticker": ticker,
        "date": last_date,
        "close": last_close,
        "change_pct": change_pct,
        "rsi": last_rsi,
        "score": score,
        "verdict": verdict,
        "signals": "；".join(signals) if signals else "无明显信号",
    }


def build_report(results, failed, report_date):
    lines = []
    lines.append(f"【规则打分】{report_date} 美股技术面评分（{len(results)} 只）")
    lines.append("")
    lines.append("方法：7 个信号投票（EMA50/EMA200/EMA20-50/RSI/MACD/位置/量价），")
    lines.append("分数 -7~+7。>=+3 偏多，<=-3 偏空（减仓参考），其余中性或弱。")
    lines.append("只看价格和成交量，不看财报和新闻；不是投资建议。")
    lines.append("")
    lines.append("| 股票 | 收盘 | 涨跌 | 规则分 | 结论 | 关键信号 |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['ticker']} | ${r['close']:.2f} | {r['change_pct']:+.2f}% | "
            f"{r['score']:+d} | {r['verdict']} | {r['signals']} |"
        )
    lines.append("")
    bull = sum(1 for r in results if r["score"] >= 3)
    bear = sum(1 for r in results if r["score"] <= -3)
    mid = len(results) - bull - bear
    lines.append(f"统计：偏多 {bull} 只，中性/弱 {mid} 只，偏空 {bear} 只。")
    if failed:
        lines.append(f"数据缺失（未计入）：{', '.join(failed)}。")
    lines.append("")
    lines.append("对照用法：和当日 LLM 仪表盘并排看。两边一致时信心高一点；")
    lines.append("两边打架（例如 LLM 说减仓、规则说偏多）时先等等，别急着动手。")
    return "\n".join(lines)


def send_email(subject, body):
    sender = os.environ.get("EMAIL_SENDER", "")
    password = os.environ.get("EMAIL_PASSWORD", "")
    receivers = os.environ.get("EMAIL_RECEIVERS", "") or sender
    if not sender or not password:
        print("未配置发件邮箱，只打印不发邮件。")
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receivers
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, [r.strip() for r in receivers.split(",") if r.strip()], msg.as_string())
    print(f"邮件已发给 {receivers}")


def main():
    tickers = [t.strip().upper() for t in
               os.environ.get("STOCK_LIST", DEFAULT_TICKERS).split(",") if t.strip()]
    results, failed = [], []
    report_date = datetime.now().strftime("%Y-%m-%d")
    for t in tickers:
        try:
            r = score_ticker(t)
            results.append(r)
            report_date = r["date"]
            print(f"{t}: ${r['close']:.2f} {r['change_pct']:+.2f}% "
                  f"RSI {r['rsi']:.0f} 分数 {r['score']:+d} {r['verdict']}")
        except Exception as e:
            failed.append(t)
            print(f"{t}: 失败（{e}）", file=sys.stderr)
    if not results:
        print("全部失败，未生成报告。", file=sys.stderr)
        return 1
    report = build_report(results, failed, report_date)
    print()
    print(report)
    with open("rules-score.md", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    send_email(f"【规则打分】{report_date} 美股技术面评分（{len(results)}只）", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
