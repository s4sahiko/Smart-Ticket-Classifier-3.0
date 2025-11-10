import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os 

# Define the log file path using the original name
LOG_FILE = os.path.join(os.getcwd(), 'usage_log.jsonl')
OUTPUT_IMAGE = 'content_gap_report.png'

def generate_report():
    """Analyzes the usage log and visualizes content gaps."""
    
    log_entries = []
    try:
        # 1.Load data from the log file
        with open(LOG_FILE, 'r') as f:
            for line in f:
                log_entries.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"\nCRITICAL ERROR: Log file '{LOG_FILE}' not found.")
        print("FIX: Ensure the Flask API (app.py) was run and you sent at least one request to /recommend.")
        return

    df = pd.DataFrame(log_entries)
    print(f"Analyzing {len(df)} usage logs...")

    # 2.Content Gap Analysis 
    gap_count = df['gap_flag'].sum()
    total_requests = len(df)
    gap_percentage = (gap_count / total_requests) * 100 if total_requests else 0
    
    print(f"\n--- Content Gap Analysis ---")
    print(f"Total recommendation requests logged: {total_requests}")
    print(f"Tickets flagged as Content Gaps (Weak/No Suggestion): {gap_count}")
    print(f"Content Gap Rate: {gap_percentage:.2f}%")
    
    if total_requests == 0:
        print("No data to visualize.")
        return

    # 3.Visualization
    plt.figure(figsize=(8, 6))
    
    df['Suggestion Status'] = df['gap_flag'].apply(
        lambda x: 'Suggestions Found' if x == 0 else 'Content Gap'
    )
    
    sns.countplot(x='Suggestion Status', data=df, palette='viridis')
    plt.title('Knowledge Base Health: Gap Detection')
    plt.xlabel('Recommendation Outcome')
    plt.ylabel('Number of Tickets')
    plt.savefig(OUTPUT_IMAGE)
    
    print(f"Dashboard visualization saved to {OUTPUT_IMAGE}")

if __name__ == '__main__':
    generate_report()