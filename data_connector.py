import gspread
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
import os

#Ensure Stopwords are downloaded
try:
    nltk.download('stopwords') 
except Exception as e:
    print(f"Warning: NLTK download failed. Text cleaning may be skipped. Error: {e}")


# Configurations

SERVICE_ACCOUNT_FILE = 'service_account.json' # File Should be in the same directory
SHEET_NAME = 'Support Tickets Data'           # CHECK THIS: Exact name of your Google Sheet
WORKSHEET_NAME = 'Sheet1'                     # CHECK THIS: Exact name of your tab in the Google Sheet
STOPWORDS = set(stopwords.words('english'))


def preprocess_text(text):
    """Cleans and standardizes a single piece of text."""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

def test_service_account_connection():
    """Tests if the service account key is found and loads successfully."""
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        print(f"\nFATAL ERROR: JSON key file '{SERVICE_ACCOUNT_FILE}' not found!")
        print("FIX: Ensure the file is named service_account.json and is in this folder.")
        return None
        
    try:
        # This line attempts to read the file and authenticate
        gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
        print("\nSUCCESS: Service Account Key loaded and authenticated.")
        return gc
    except Exception as e:
        print(f"\nFATAL ERROR: Service Account failed to authenticate. FIX THIS ISSUE:\n{e}")
        print("FIX: The key file is corrupted or not configured correctly.")
        return None

def fetch_and_preprocess_data(gc):
    """Fetches data using the authenticated client and cleans it."""
    try:
        # 1. Connect to the specific sheet and worksheet
        spreadsheet = gc.open(SHEET_NAME)
        worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
        
        # 2. Get data
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
        print(f"Successfully fetched {len(df)} rows of raw ticket data.")
        
        # 3. Preprocessing Logic
        df['Full_Ticket_Text'] = df['Subject'].astype(str) + " " + df['Description'].astype(str)
        df['Clean_Ticket_Text'] = df['Full_Ticket_Text'].apply(preprocess_text)
        
        # 4. Save the critical file (Milestone 1 Deliverable)
        df.to_csv('cleaned_ticket_data.csv', index=False)
        print("Preprocessing complete. Data saved to 'cleaned_ticket_data.csv'.")
        return df
        
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"\nCRITICAL FAILURE: SpreadsheetNotFound! Check that SHEET_NAME ('{SHEET_NAME}') is an EXACT match to your Google Sheet title.")
        print("FIX: Copy the name directly from the Google Sheet title bar.")
    except gspread.exceptions.WorksheetNotFound:
        print(f"\nCRITICAL FAILURE: WorksheetNotFound! Check that WORKSHEET_NAME ('{WORKSHEET_NAME}') is an EXACT match to your tab name.")
    except gspread.exceptions.APIError as e:
        # This usually means Permission Denied
        print(f"\nCRITICAL FAILURE: APIError/Permission Denied! FIX THIS ISSUE:\n{e}")
        print("FIX: You must SHARE your Google Sheet with the client email address found inside 'service_account.json'.")
    except Exception as e:
        print(f"\nAN UNKNOWN ERROR OCCURRED during fetching or preprocessing: {e}")
        
    return pd.DataFrame()


if __name__ == '__main__':
    # 1. Test the key file first. If it works, proceed.
    gc = test_service_account_connection()
    
    if gc is not None:
        # 2. Key works, now try to fetch the data
        df = fetch_and_preprocess_data(gc)
        
        if not df.empty:
            print("\nSUCCESS: Data fetching and preprocessing completed without critical errors.")
        else:
            print("\nCANNOT PROCEED: Review the CRITICAL FAILURE message above and apply the fix.")
    else:
        print("\nCANNOT PROCEED: Fix the JSON Key error first.")