import yfinance as yf
import logging
from datetime import datetime
from typing import Dict, List
from bs4 import BeautifulSoup
import requests
import time
from snowflake_operations import SnowflakeOperations
import schedule
import sys
import traceback


# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("stock_news_sync.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Magnificent 7 stocks
MAG7_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
# MAG7_STOCKS = ["AAPL"]
snowflake_ops = SnowflakeOperations()


def get_article_content(url: str) -> str:
    """
    Fetch article content from URL with retry mechanism and logging.
    """
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            content = " ".join(
                [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
            )

            # Only consider content if it's substantial (more than 500 characters)
            if content and len(content) > 500:
                logger.info(f"Successfully fetched content from {url}")
                return content
            else:
                logger.warning(f"Insufficient content found at {url}")
                return ""

        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {str(e)}")
            return ""

    logger.error(f"Failed to fetch content after {max_retries} attempts for {url}")
    return ""


def fetch_stock_news(symbol: str, cutoff_date: datetime) -> List[Dict]:
    """
    Fetch news for a given stock symbol using yfinance

    Args:
        symbol: Stock symbol
        cutoff_date: Datetime to filter articles from
    """
    logger.info(f"Fetching news for {symbol} since {cutoff_date}")
    try:
        news = yf.Search(symbol, news_count=20).news

        processed_news = []
        articles_found = len(news)

        logger.info(f"Found {articles_found} total articles for {symbol}")
        skipped_count = 0
        processed_count = 0

        for article in news:
            try:
                pub_date = datetime.fromtimestamp(article.get("providerPublishTime", 0))
                title = article.get("title", "").strip()

                if not title:
                    logger.warning("Skipping article with empty title")
                    skipped_count += 1
                    continue

                if pub_date >= cutoff_date:
                    url = article.get("link", "")
                    logger.debug(f"Processing article: {title[:100]}...")

                    # use Yahoo uuid to find unique
                    article_id = article["uuid"]

                    content = get_article_content(url)
                    if content:
                        processed_news.append(
                            {
                                "article_id": article_id,
                                "symbol": symbol,
                                "title": title,
                                "publication_date": pub_date,
                                "source": article.get("publisher", ""),
                                "content": content,
                                "link": url,
                            }
                        )
                        processed_count += 1
                        logger.debug(
                            f"Successfully processed article: {title[:100]} (ID: {article_id})"
                        )
                    else:
                        logger.warning(
                            f"Skipped article due to empty content: {title[:100]}"
                        )
                        skipped_count += 1
                else:
                    logger.debug(
                        f"Skipped article due to date: {title[:100]} ({pub_date})"
                    )
                    skipped_count += 1

            except Exception as e:
                logger.error(f"Error processing individual article: {str(e)}")
                skipped_count += 1
                continue

        logger.info(
            f"News processing completed for {symbol}:\n"
            f"- Total articles found: {articles_found}\n"
            f"- Successfully processed: {processed_count}\n"
            f"- Skipped articles: {skipped_count}"
        )
        return processed_news

    except Exception as e:
        logger.error(
            f"Error fetching news for {symbol}: {str(e)}\n{traceback.format_exc()}"
        )
        return []


def sync_news():
    """
    Main function to sync news for all Magnificent 7 stocks
    """
    logger.info("Starting news sync job")
    start_time = time.time()

    try:
        snowflake_ops = SnowflakeOperations()
        total_articles = 0
        failed_symbols = []

        # Get last run timestamp
        last_run = snowflake_ops.get_last_run("news_sync")
        last_run = last_run.replace(tzinfo=None)

        logger.info(f"Last successful run timestamp: {last_run}")

        for symbol in MAG7_STOCKS:
            try:
                logger.info(f"Processing {symbol}")
                articles = fetch_stock_news(symbol, last_run)

                if articles:
                    snowflake_ops.insert_news_articles(articles)
                    total_articles += len(articles)
                    logger.info(
                        f"Successfully stored {len(articles)} articles for {symbol}"
                    )
                else:
                    logger.info(f"No new articles found for {symbol}")

                # Delay between stocks to avoid rate limiting
                time.sleep(2)

            except Exception as e:
                logger.error(f"Failed to process {symbol}: {str(e)}")
                failed_symbols.append(symbol)
                continue

        # Update last run timestamp only if articles were processed
        if total_articles > 0:
            snowflake_ops.update_last_run("news_sync", "completed", total_articles)
            logger.info("Updated last run timestamp")

        # Log summary
        execution_time = time.time() - start_time
        logger.info(
            f"""
        News sync completed:
        - Total articles stored: {total_articles}
        - Failed symbols: {', '.join(failed_symbols) if failed_symbols else 'None'}
        - Execution time: {execution_time:.2f} seconds
        """
        )

    except Exception as e:
        logger.error(f"Critical error in sync job: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)


def run_scheduled_job():
    """
    Run the sync job on a schedule
    """
    logger.info("Starting scheduled news sync job")

    # Schedule job to run every 6 hours
    schedule.every(6).hours.do(sync_news)

    # Run immediately on start
    sync_news()
    snowflake_ops.build_incremental_graph()

    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check schedule every minute


if __name__ == "__main__":
    # Check if running as one-time job or scheduled service
    if len(sys.argv) > 1 and sys.argv[1] == "--schedule":
        run_scheduled_job()
    else:
        sync_news()
        snowflake_ops.build_incremental_graph()
