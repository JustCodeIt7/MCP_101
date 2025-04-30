from mcp.server.fastmcp import FastMCP
import yfinance as yf
from newsapi import NewsApiClient
from textblob import TextBlob
import os
import time
from datetime import datetime, timedelta
import json

# Instantiate FastMCP
mcp = FastMCP("stock_insights")

# Initialize NewsAPI client with a placeholder API key
# In a real implementation, you would use your actual NewsAPI key
# Get a free API key at https://newsapi.org/
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "your_news_api_key_here")
news_api = NewsApiClient(api_key=NEWS_API_KEY)

# Simple cache to avoid hitting rate limits
cache = {}
CACHE_EXPIRY = 300  # 5 minutes in seconds

@mcp.tool()
def get_stock_insights(ticker: str) -> dict:
    """
    Get stock price and sentiment analysis for a given ticker symbol.
    
    Args:
        ticker: The stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
        
    Returns:
        A dictionary containing the current price, recent headlines, and sentiment analysis
    """
    ticker = ticker.upper()
    
    # Check cache first
    if ticker in cache and time.time() - cache[ticker]["timestamp"] < CACHE_EXPIRY:
        return cache[ticker]["data"]
    
    try:
        # Get stock price using yfinance
        price_data = get_stock_price(ticker)
        
        # Get news headlines
        headlines, news_urls = get_news_headlines(ticker)
        
        # Perform sentiment analysis
        sentiment_scores = analyze_sentiment(headlines)
        
        # Prepare the result
        result = {
            "ticker": ticker,
            "price_data": price_data,
            "headlines": headlines[:5],  # Limit to 5 headlines
            "news_urls": news_urls[:5],  # Limit to 5 URLs
            "sentiment": sentiment_scores,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Cache the result
        cache[ticker] = {
            "timestamp": time.time(),
            "data": result
        }
        
        return result
    
    except Exception as e:
        return {
            "ticker": ticker,
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def get_stock_price(ticker):
    """Get stock price data using yfinance."""
    try:
        # Get stock info
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant price data
        price_data = {
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", None)),
            "previous_close": info.get("previousClose", None),
            "open": info.get("open", None),
            "day_high": info.get("dayHigh", None),
            "day_low": info.get("dayLow", None),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh", None),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow", None),
            "market_cap": info.get("marketCap", None),
            "volume": info.get("volume", None)
        }
        
        # If we couldn't get the current price, try to get it from history
        if price_data["current_price"] is None:
            history = stock.history(period="1d")
            if not history.empty:
                price_data["current_price"] = history["Close"].iloc[-1]
        
        return price_data
    
    except Exception as e:
        # If there's an error with yfinance, simulate some data for demonstration
        print(f"Error getting stock price: {str(e)}")
        return simulate_stock_data(ticker)

def get_news_headlines(ticker):
    """Get news headlines using NewsAPI."""
    try:
        if NEWS_API_KEY == "your_news_api_key_here":
            # If no API key is provided, use simulated data
            return simulate_news_headlines(ticker)
        
        # Calculate date range (last 7 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Format dates for NewsAPI
        from_date = start_date.strftime("%Y-%m-%d")
        to_date = end_date.strftime("%Y-%m-%d")
        
        # Get news articles
        response = news_api.get_everything(
            q=f"{ticker} stock",
            from_param=from_date,
            to=to_date,
            language="en",
            sort_by="relevancy",
            page_size=10
        )
        
        # Extract headlines and URLs
        headlines = []
        urls = []
        
        if response["status"] == "ok":
            articles = response["articles"]
            for article in articles:
                headlines.append(article["title"])
                urls.append(article["url"])
        
        return headlines, urls
    
    except Exception as e:
        print(f"Error getting news headlines: {str(e)}")
        return simulate_news_headlines(ticker)

def analyze_sentiment(headlines):
    """Analyze sentiment of headlines using TextBlob."""
    if not headlines:
        return {
            "average_polarity": 0,
            "average_subjectivity": 0,
            "sentiment_label": "Neutral",
            "headline_sentiments": []
        }
    
    total_polarity = 0
    total_subjectivity = 0
    headline_sentiments = []
    
    for headline in headlines:
        blob = TextBlob(headline)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment label
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        headline_sentiments.append({
            "headline": headline,
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": sentiment
        })
        
        total_polarity += polarity
        total_subjectivity += subjectivity
    
    # Calculate averages
    avg_polarity = total_polarity / len(headlines)
    avg_subjectivity = total_subjectivity / len(headlines)
    
    # Determine overall sentiment
    if avg_polarity > 0.1:
        overall_sentiment = "Positive"
    elif avg_polarity < -0.1:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"
    
    return {
        "average_polarity": avg_polarity,
        "average_subjectivity": avg_subjectivity,
        "sentiment_label": overall_sentiment,
        "headline_sentiments": headline_sentiments
    }

def simulate_stock_data(ticker):
    """Simulate stock data for demonstration purposes."""
    import random
    
    base_price = hash(ticker) % 1000 + 50  # Generate a pseudo-random base price
    
    # Add some randomness
    current_price = base_price * (1 + random.uniform(-0.05, 0.05))
    previous_close = current_price * (1 + random.uniform(-0.02, 0.02))
    day_open = previous_close * (1 + random.uniform(-0.01, 0.01))
    day_high = current_price * (1 + random.uniform(0, 0.02))
    day_low = current_price * (1 + random.uniform(-0.02, 0))
    
    return {
        "current_price": round(current_price, 2),
        "previous_close": round(previous_close, 2),
        "open": round(day_open, 2),
        "day_high": round(day_high, 2),
        "day_low": round(day_low, 2),
        "fifty_two_week_high": round(current_price * 1.2, 2),
        "fifty_two_week_low": round(current_price * 0.8, 2),
        "market_cap": round(current_price * 1000000000, 2),
        "volume": random.randint(1000000, 10000000),
        "simulated": True
    }

def simulate_news_headlines(ticker):
    """Simulate news headlines for demonstration purposes."""
    headlines = [
        f"{ticker} Reports Strong Quarterly Earnings, Exceeding Analyst Expectations",
        f"Investors Optimistic About {ticker}'s New Product Launch",
        f"{ticker} Announces Strategic Partnership to Expand Market Reach",
        f"Analysts Upgrade {ticker} Stock Rating to 'Buy'",
        f"{ticker} Faces Challenges in Global Supply Chain",
        f"CEO of {ticker} Discusses Future Growth Strategies in Recent Interview",
        f"{ticker} Stock Fluctuates Amid Market Volatility",
        f"Industry Experts Predict Positive Outlook for {ticker}",
        f"{ticker} Implements Cost-Cutting Measures to Improve Profitability",
        f"New Regulations Could Impact {ticker}'s Business Model"
    ]
    
    # Shuffle the headlines to add randomness
    import random
    random.shuffle(headlines)
    
    # Generate fake URLs
    urls = [f"https://example.com/news/{ticker.lower()}/{i}" for i in range(len(headlines))]
    
    return headlines, urls

@mcp.tool()
def list_popular_tickers() -> list:
    """
    Get a list of popular stock ticker symbols.
    
    Returns:
        A list of popular ticker symbols
    """
    return [
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL", # Alphabet (Google)
        "AMZN",  # Amazon
        "META",  # Meta (Facebook)
        "TSLA",  # Tesla
        "NVDA",  # NVIDIA
        "JPM",   # JPMorgan Chase
        "V",     # Visa
        "WMT",   # Walmart
        "JNJ",   # Johnson & Johnson
        "PG",    # Procter & Gamble
        "DIS",   # Disney
        "NFLX",  # Netflix
        "INTC"   # Intel
    ]

if __name__ == "__main__":
    mcp.run(transport="stdio")