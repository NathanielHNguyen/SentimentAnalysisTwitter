import tweepy
import pandas as pd

# Twitter API credentials
bearer_token = "AAAAAAAAAAAAAAAAAAAAADL4xQEAAAAAFO7iHhnR9dWIHTqtOLVZPFARcV4%3DOWyCEiEXhGRpYPfiM3CIp6nqYjC32NVvTp4iy65ETwqG2HTAMP"

# Authenticate with the Twitter API using Bearer Token
client = tweepy.Client(bearer_token=bearer_token)

query = "UNH"
max_results = 50  

# List to hold the scraped tweets
data = []

print(f"Scraping tweets containing the keyword '{query}'...")
response = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=["created_at", "text", "author_id"])

if response.data:
    for tweet in response.data:
        data.append([
            tweet.created_at,
            tweet.author_id,
            tweet.text,
            f"https://twitter.com/i/web/status/{tweet.id}"
        ])

# Convert the scraped data to a DataFrame
df = pd.DataFrame(data, columns=["Date", "Author ID", "Content", "URL"])

# Save the data to a CSV file
output_file = "tweets_with_x.csv"
df.to_csv(output_file, index=False)

print(f"Scraping complete! {len(df)} tweets saved to '{output_file}'.")
