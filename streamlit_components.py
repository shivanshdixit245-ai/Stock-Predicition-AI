import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_candlestick(df: pd.DataFrame, signals: pd.Series = None, regimes: pd.Series = None) -> go.Figure:
    """
    Plots a professional candlestick chart with indicators and buy/sell signals.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('Price', 'Volume'), 
                        row_width=[0.2, 0.7])

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'
    ), row=1, col=1)

    # 2. Indicators (SMA20/50, Bollinger)
    if 'sma_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], line=dict(color='orange', width=1), name='SMA 20'), row=1, col=1)
    if 'sma_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], line=dict(color='blue', width=1), name='SMA 50'), row=1, col=1)
    
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], line=dict(color='rgba(173, 216, 230, 0.4)', width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], line=dict(color='rgba(173, 216, 230, 0.4)', width=0), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.1)', name='Bollinger Bands'), row=1, col=1)

    # 3. Signals
    if signals is not None:
        buy_signals = df[signals == 1]
        sell_signals = df[signals == -1]
        
        fig.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals['low'] * 0.99,
            mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
            name='BUY'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=sell_signals.index, y=sell_signals['high'] * 1.01,
            mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
            name='SELL'
        ), row=1, col=1)

    # 4. Volume
    colors = ['red' if row['open'] - row['close'] > 0 else 'green' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='Volume'), row=2, col=1)

    # 5. Regime Shading (Optimized)
    if regimes is not None:
        regime_colors = {0: 'rgba(0, 255, 0, 0.05)', 1: 'rgba(255, 0, 0, 0.05)', 2: 'rgba(128, 128, 128, 0.05)'}
        
        # Group contiguous regimes for efficiency
        # Handle index alignment: ensure we only use indices present in both
        common_idx = df.index.intersection(regimes.index)
        local_regimes = regimes.loc[common_idx]
        
        if len(local_regimes) > 0:
            diff = local_regimes != local_regimes.shift(1)
            change_points = local_regimes.index[diff]
            
            for i in range(len(change_points)):
                start_idx = change_points[i]
                end_idx = change_points[i+1] if i+1 < len(change_points) else local_regimes.index[-1]
                regime_val = local_regimes.loc[start_idx]
                
                fig.add_vrect(
                    x0=start_idx, x1=end_idx,
                    fillcolor=regime_colors.get(regime_val, 'grey'),
                    layer='below', line_width=0, row=1, col=1
                )

    fig.update_layout(height=800, xaxis_rangeslider_visible=False, template='plotly_white')
    return fig

def plot_equity_comparison(results: dict, benchmarks: dict) -> go.Figure:
    """
    Plots the strategy equity curve against multiple benchmarks with drawdown shading.
    """
    fig = go.Figure()
    
    # Strategy curve
    strategy_curve = results['equity_curve']
    fig.add_trace(go.Scatter(x=strategy_curve.index, y=strategy_curve, line=dict(color='#534AB7', width=2), name='ML Strategy'))
    
    # Benchmarks
    for name, curve in benchmarks.items():
        dash = 'dash' if name == 'Buy & Hold' else 'dot'
        color = 'grey' if name == 'Buy & Hold' else 'teal'
        fig.add_trace(go.Scatter(x=curve.index, y=curve, line=dict(color=color, dash=dash, width=1.5), name=name))
    
    # Drawdown shading for strategy
    dd = results['drawdown_curve']
    fig.add_trace(go.Scatter(
        x=dd.index, y=strategy_curve * (1 + dd),
        fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(width=0), showlegend=False, name='Drawdown'
    ))
    
    fig.update_layout(title='Performance Comparison', yaxis_title='Portfolio Value', template='plotly_white', height=500)
    return fig

def plot_bootstrap_hist(bootstrap_result: dict) -> go.Figure:
    """
    Plots the histogram of bootstrap Sharpe ratios with the observed value.
    """
    null_sharpes = bootstrap_result['null_sharpes']
    observed = bootstrap_result['observed_sharpe']
    p_value = bootstrap_result['p_value']
    
    fig = px.histogram(null_sharpes, nbins=50, labels={'value': 'Sharpe Ratio'}, title=f'Bootstrap Significance Test (p={p_value:.4f})')
    fig.add_vline(x=observed, line_dash="dash", line_color="#534AB7", annotation_text="Observed Sharpe")
    
    # Shade 95th percentile
    percentile_95 = np.percentile(null_sharpes, 95)
    fig.add_vrect(x0=percentile_95, x1=max(null_sharpes), fillcolor="green", opacity=0.1, layer="below", line_width=0)
    
    fig.update_layout(template='plotly_white')
    return fig

def plot_shap_waterfall(shap_values, feature_names) -> go.Figure:
    """
    Plots a custom Plotly waterfall for SHAP values instead of using matplotlib.
    """
    # Simply showing top 15 features as a bar chart if waterfall is too complex for session state
    # But requested waterfall. We'll use a horizontal bar chart with color coding for +/- impact.
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': shap_values
    }).sort_values('importance', ascending=True).tail(15)
    
    fig = go.Figure(go.Scatter(
        x=df['importance'],
        y=df['feature'],
        mode='markers+text',
        marker=dict(color=['red' if val < 0 else 'green' for val in df['importance']], size=12),
        text=df['importance'].apply(lambda x: f"{x:.4f}"),
        textposition="middle right"
    ))
    
    fig.update_layout(title='SHAP Feature Impact (Latest Prediction)', xaxis_title='SHAP Value', template='plotly_white', height=500)
    return fig

def plot_regime_timeline(df: pd.DataFrame, regimes: pd.Series) -> go.Figure:
    """
    Plots the price chart with color-coded regime segments (scatter-based, no vrects).
    """
    fig = go.Figure()
    
    regime_colors = {0: 'green', 1: 'red', 2: 'grey'}
    regime_names = {0: 'Bull', 1: 'Bear', 2: 'Sideways'}
    
    # Align indices
    common_idx = df.index.intersection(regimes.index)
    aligned_df = df.loc[common_idx]
    aligned_regimes = regimes.loc[common_idx]
    
    # Plot each regime as a separate colored scatter trace
    for regime_id in sorted(aligned_regimes.unique()):
        mask = aligned_regimes == regime_id
        regime_df = aligned_df[mask]
        fig.add_trace(go.Scatter(
            x=regime_df.index, 
            y=regime_df['close'],
            mode='markers+lines',
            marker=dict(color=regime_colors.get(regime_id, 'grey'), size=3),
            line=dict(color=regime_colors.get(regime_id, 'grey'), width=1.5),
            name=regime_names.get(regime_id, f'State {regime_id}'),
            connectgaps=False
        ))

    fig.update_layout(title='Market Regimes Timeline', template='plotly_white', height=500)
    return fig


def render_signal_badge(signal: str) -> str:
    """
    Returns HTML for a styled signal badge.
    """
    color = "green" if signal == "BUY" else "red" if signal == "SELL" else "grey"
    return f'<span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">{signal}</span>'
