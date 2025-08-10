import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load original file
df = pd.read_csv("data/simulated_customers.csv")

# Add SignupDate: simulate based on LastPurchaseDate
df['LastPurchaseDate'] = pd.to_datetime(df['LastPurchaseDate'])
df['SignupDate'] = df['LastPurchaseDate'] - pd.to_timedelta(np.random.randint(180, 720, size=len(df)), unit='D')
df['SignupDate'] = df['SignupDate'].dt.strftime('%Y-%m-%d')

# Add LoginFrequency: simulate logins per month
df['LoginFrequency'] = np.random.randint(1, 31, size=len(df))

# Add EngagementScore: based on usage (can be tuned further)
df['EngagementScore'] = (
    (df['LoginFrequency'] * 0.4) +
    (df['NumTransactions'] * 0.3) +
    (df['TotalSpend'] / 100 * 0.3)
).round(2)

# Save updated file (overwrite original)
df.to_csv("data/simulated_customers.csv", index=False)

print("✅ Dataset enhanced and saved successfully!")
