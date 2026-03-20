"""
AI Stock Assistant — Gemini-powered conversational analysis.

Powers an AI chat assistant that answers any question about the currently
selected stock using Google Gemini-2.0-flash via the Generative AI API.
The assistant has full context about the stock including current signals,
backtest results, regime, stress score, and disaster forecast.

DS Interview Note:
Switching to Gemini-2.0-flash provides a balance of high performance and
favourable rate limits (free tier). The system prompt engineering ensures
the model stays factual, cites actual model metrics, and never overstates
confidence.
"""

import os
from datetime import datetime
from loguru import logger
import pandas as pd
import google.generativeai as genai
from config import settings

# ============================================================================
# 1. Build Stock Context
# ============================================================================

def build_stock_context(
    ticker: str,
    price_df: pd.DataFrame,
    signals: pd.Series,
    backtest_results: dict,
    regime: str,
    stress_status: dict,
    disaster_forecast: dict,
    walk_forward_metrics: dict,
) -> str:
    """
    Build a structured text summary of everything the model knows about the
    current ticker, to be injected into the AI system prompt.

    DS Interview Note:
    Context engineering is the most important part of LLM integration.
    Giving the model access to *actual numbers* from the pipeline prevents
    hallucination. Every claim the AI makes can be traced back to a
    specific metric in this context block.
    """
    try:
        latest_price = float(price_df["close"].iloc[-1])
        prev_price = float(price_df["close"].iloc[-2]) if len(price_df) > 1 else latest_price
        pct_change = (latest_price - prev_price) / prev_price * 100

        latest_signal = str(signals.iloc[-1]) if len(signals) > 0 else "N/A"
        buy_prob = float(price_df["buy_prob"].iloc[-1]) if "buy_prob" in price_df.columns else 0.0

        # Recent signals table
        recent_n = min(10, len(price_df))
        recent = price_df.tail(recent_n)
        recent_lines = []
        for idx, row in recent.iterrows():
            sig = row.get("signal", "N/A")
            prob = row.get("buy_prob", 0)
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
            recent_lines.append(f"  {date_str} | {sig} | {prob:.2f}")
        recent_table = "\n".join(recent_lines)

        # Backtest metrics
        bt_metrics = backtest_results.get("metrics", {})
        sharpe = bt_metrics.get("sharpe_ratio", "N/A")
        total_return = bt_metrics.get("total_return", "N/A")
        max_dd = bt_metrics.get("max_drawdown", "N/A")

        # Walk-forward
        wf_f1 = walk_forward_metrics.get("f1", "N/A")
        wf_acc = walk_forward_metrics.get("accuracy", "N/A")

        # Stress & disaster
        stress_score = stress_status.get("stress_score", 0)
        stress_label = stress_status.get("label", "Normal")
        stress_action = stress_status.get("action", "trade_normally")
        circuit_breaker_active = stress_action == "circuit_breaker"

        ews = disaster_forecast.get("early_warning_score", 0)
        prob_5d = disaster_forecast.get("prob_disaster_5d", 0)
        prob_10d = disaster_forecast.get("prob_disaster_10d", 0)
        prob_30d = disaster_forecast.get("prob_disaster_30d", 0)
        forecast_confidence = disaster_forecast.get("confidence", "low")

        # Build context
        ctx = f"""
=== CURRENT STATE ===
Ticker: {ticker}
Latest Price: ${latest_price:.2f} ({pct_change:+.2f}% today)
Current Signal: {latest_signal} (buy probability: {buy_prob:.1%})
Current Market Regime: {regime}
Stress Score: {stress_score:.0f}/100 ({stress_label})
Stress Action: {stress_action}
Circuit Breaker Active: {"YES — ALL SIGNALS OVERRIDDEN TO HOLD" if circuit_breaker_active else "No"}
Early Warning Score: {ews:.0f}/100
Disaster Probability: 5d={prob_5d:.0%}, 10d={prob_10d:.0%}, 30d={prob_30d:.0%}
Forecast Confidence: {forecast_confidence.upper()}

=== MODEL PERFORMANCE ===
Walk-Forward F1: {wf_f1}
Walk-Forward Accuracy: {wf_acc}
Sharpe Ratio: {sharpe}
Total Return: {total_return}
Max Drawdown: {max_dd}

=== RECENT SIGNALS (last {recent_n} days) ===
  Date       | Signal | Buy Prob
{recent_table}

=== RISK WARNINGS ==="""

        warnings = []
        if circuit_breaker_active:
            warnings.append("🚨 CIRCUIT BREAKER IS ACTIVE — extreme market stress detected. All model signals have been overridden to HOLD.")
        if prob_5d > 0.6:
            top_indicator = disaster_forecast.get("top_3_warning_indicators", ["unknown"])[0]
            warnings.append(f"⚠️ EARLY WARNING: {prob_5d:.0%} probability of market stress in next 5 days. Top signal: {top_indicator}")
        if forecast_confidence == "low":
            warnings.append("⚠️ Model confidence is LOW — insufficient historical data for reliable forecasting.")

        if warnings:
            ctx += "\n" + "\n".join(warnings)
        else:
            ctx += "\nNo active risk warnings."

        logger.debug(f"Built stock context for {ticker} ({len(ctx)} chars)")
        return ctx.strip()

    except Exception as e:
        logger.error(f"Failed to build stock context: {e}")
        return f"Limited context available for {ticker}. Error: {e}"


# ============================================================================
# 2. Build System Prompt
# ============================================================================

def build_system_prompt(stock_context: str) -> str:
    """
    Returns the full system prompt for Gemini with analysis context injected.

    DS Interview Note:
    The system prompt is the guardrail. It constrains the model to (1) cite
    actual model outputs, (2) never give financial advice, (3) always
    disclose model confidence, and (4) automatically mention circuit
    breakers when active.
    """
    prompt = f"""You are an expert stock analysis assistant with deep knowledge of quantitative finance, machine learning, and technical analysis.

You have access to a sophisticated AI-powered analysis system that has analyzed stocks using ensemble ML, walk-forward cross-validation, conformal prediction, and market regime detection.

Here is the complete current analysis:

{stock_context}

Your role:
1. Answer questions about this stock using the provided data
2. Explain what the signals, indicators and metrics mean in plain language that a retail investor can understand
3. Explain the model's reasoning using SHAP feature importance
4. ALWAYS mention model confidence and limitations
5. NEVER give direct financial advice or say "buy this stock"
6. ALWAYS add: "This is for educational purposes only, not financial advice."
7. If asked about future disasters or crashes, refer to the Early Warning System probabilities but emphasize uncertainty
8. If the model has low confidence, say so clearly
9. Be specific — use the actual numbers from the context
10. Keep responses concise — under 200 words unless asked to elaborate

HIGH CONFIDENCE RULE:
Only make claims supported by the model data.
If the model F1 < 0.55 say: "The model has limited confidence for this ticker."
If bootstrap p-value > 0.05 say: "The strategy alpha is not statistically significant for this ticker."
Never overstate model certainty.

If the CIRCUIT BREAKER is active, you MUST mention it prominently in every response and explain that all signals are overridden to HOLD due to extreme market conditions."""

    logger.debug(f"System prompt built ({len(prompt)} chars)")
    return prompt


# ============================================================================
# 3. Ask Assistant (Google Gemini API)
# ============================================================================

def ask_assistant(
    user_question: str,
    stock_context: str,
    conversation_history: list
) -> str:
    import google.generativeai as genai
    from config import settings
    from loguru import logger

    key = settings.gemini_api_key
    logger.info(f"Gemini key present: {bool(key)} | length: {len(key) if key else 0}")

    if not key or len(key.strip()) < 10:
        return (
            "AI Assistant not configured. "
            "Get your FREE Gemini key at: "
            "https://aistudio.google.com/apikey "
            "Then add to your .env file: "
            "GEMINI_API_KEY=your_key_here "
            "Then restart Streamlit with: "
            "python -m streamlit run app.py"
        )

    try:
        genai.configure(api_key=key.strip())

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=build_system_prompt(stock_context)
        )

        history = []
        for msg in conversation_history[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        chat = model.start_chat(history=history)

        response = chat.send_message(
            user_question,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=500,
            )
        )
        return response.text

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Gemini API error: {type(e).__name__}: {error_msg}")
        
        # Specific handling for Rate Limits (429)
        if "429" in error_msg or "quota" in error_msg.lower():
            return (
                "⚠️ AI Assistant Rate Limit Reached (429). \n\n"
                "The Gemini Free Tier allows 15 requests per minute. "
                "Please wait 30-60 seconds and try again. \n\n"
                "If you need higher limits, consider using a paid tier or "
                "entering a different API key in your .env.LOCAL file."
            )
        
        # Clean up generic errors (remove link soup)
        clean_error = error_msg.split(" - ")[0].split("details:")[0].strip()
        return f"AI assistant error: {clean_error if clean_error else 'Unknown error'}"


# ============================================================================
# 4. Format Response with Confidence
# ============================================================================

def format_response_with_confidence(response: str, context: dict) -> dict:
    """
    Wrap the raw AI response with confidence metadata and disclaimer.
    """
    # 0. Handle Error Strings (bypass scoring)
    if response.startswith("⚠️") or response.startswith("AI assistant error"):
        return {
            "response_text": response,
            "confidence_level": "N/A",
            "disclaimer": "SERVICE_ERROR",
            "data_freshness": datetime.now().strftime("%Y-%m-%d %H:%M")
        }

    # 1. Determine confidence from model metrics
    f1 = context.get("f1", 0)
    p_value = context.get("bootstrap_p_value", 1.0)

    if f1 >= 0.65 and p_value < 0.05:
        confidence = "HIGH"
    elif f1 >= 0.55:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    result = {
        "response_text": response,
        "confidence_level": confidence,
        "disclaimer": "Educational purposes only. Not financial advice.",
        "data_freshness": context.get("data_freshness", datetime.now().strftime("%Y-%m-%d %H:%M")),
    }

    logger.debug(f"Formatted response with confidence={confidence}")
    return result
