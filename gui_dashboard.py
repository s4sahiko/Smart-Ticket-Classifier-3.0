import streamlit as st
import requests
import pandas as pd
import random
import os
import io
import json
import matplotlib.pyplot as plt 
import seaborn as sns           
import plotly.express as px # ADDED: For interactive charts in Analytics section


# --- Configuration ---
# API_URL is correctly set to 5000 to match your Flask backend.
API_URL = "http://127.0.0.1:5000/recommend"
# NEW: Endpoint for fetching real-time Slack chat activity
SLACK_API_URL = "http://127.0.0.1:5000/slack_activity" 
LOG_FILE_PATH = os.path.join(os.getcwd(), 'usage_log.jsonl') 
REPORT_IMAGE_PATH = 'content_gap_report.png'

# --- Streamlit Session State Initialization ---
if 'show_analytics' not in st.session_state:
    st.session_state['show_analytics'] = False
if 'all_results_df' not in st.session_state:
    st.session_state['all_results_df'] = pd.DataFrame()
# NEW: Notification system
if 'notifications' not in st.session_state:
    st.session_state['notifications'] = []


# --- Helper Functions ---

def get_predictions(ticket_text):
    """
    Sends the ticket text to the Flask API and returns ALL prediction data, 
    including the new 'slack_sent' status.
    """
    if not ticket_text:
        # UPDATED: Must return 5 values now
        return None, None, None, None, False 
    
    # Ensure a unique ID for each API call
    ticket_id = f"GUI-{random.randint(1000, 9999)}"
    
    payload = {
        "ticket_id": ticket_id,
        "ticket_text": ticket_text
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() 
        
        data = response.json()
        
        # UPDATED: Return 5 values, including 'slack_sent'
        return (
            data.get('suggestions', []), 
            data.get('severity_prediction', None),
            data.get('issue_prediction', None),
            data.get('team_prediction', None),
            data.get('slack_sent', False) # NEW: Capture Slack status
        )
        
    except requests.exceptions.ConnectionError:
        st.error("API Connection failed. Please ensure your backend (app.py) is running on port 5000.")
        return None, None, None, None, False
    except requests.exceptions.RequestException as e:
        st.error(f"API Request failed: {e}")
        return None, None, None, None, False


def display_prediction_score(prediction_info):
    """Formats the prediction label and score for consistent display."""
    if not prediction_info:
        label = "N/A"
        return {"label": label, "score_text": "0.0%"}
    
    label = prediction_info.get('label', "N/A")
    score = prediction_info.get('score', 0.0)
    
    return {"label": label, "score_text": f"{score * 100:.1f}%"}


def extract_text_from_upload(uploaded_file):
    """Extracts text content from uploaded files."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'txt':
        try:
            raw_content = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            ticket_blocks = raw_content.split('\n\n\n')
            
            tickets_list = []
            for i, block in enumerate(ticket_blocks):
                cleaned_text = block.replace('\n', ' ').strip()
                if cleaned_text:
                    tickets_list.append({'text': cleaned_text, 'source': f"TXT Bulk Ticket {i + 1}"})

            if not tickets_list:
                st.warning("TXT file contains no tickets, or they are not separated by three empty lines.")
                return []
            st.success(f"Extracted {len(tickets_list)} tickets from TXT file using the 'three empty lines' separator.")
            return tickets_list
        
        except Exception as e:
            st.error(f"Error processing bulk TXT file: {e}")
            return []
            
    elif file_extension == 'csv':
        try:
            df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
            text_columns = ['Description', 'Subject', 'Full_Ticket_Text', 'text', 'ticket_text']
            ticket_column = next((col for col in text_columns if col in df.columns), None)
            
            if ticket_column:
                tickets_list = []
                for index, row in df.iterrows():
                    raw_text = row[ticket_column]
                    if pd.notna(raw_text) and str(raw_text).strip():
                        cleaned_text = str(raw_text).replace('\n', ' ').strip()
                        tickets_list.append({'text': cleaned_text, 'source': f"CSV Row {index + 1}"})

                st.success(f"Extracted {len(tickets_list)} clean tickets from CSV using column: '{ticket_column}'.")
                return tickets_list
            else:
                st.error("Could not find a suitable text column (e.g., 'Description', 'Subject') in the CSV.")
                return []

        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return []
            
    else:
        st.error(f"Unsupported file format: .{file_extension}. Only .txt and .csv are supported.")
        return []


def generate_gap_report():
    """Reads the log file, generates the Content Gap Analysis chart, and returns key metrics."""
    default_metrics = {"message": "No data logged yet."}
    
    if not os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'w') as f: pass
        return False, None, default_metrics
    
    try:
        data = []
        with open(LOG_FILE_PATH, 'r') as f:
            for line in f:
                try: data.append(json.loads(line))
                except json.JSONDecodeError: continue 

        if not data: 
            return False, None, default_metrics

        df_log = pd.DataFrame(data)
        
        gap_count = df_log['gap_flag'].sum()
        total_count = len(df_log)
        successful_matches = total_count - gap_count
        gap_ratio = gap_count / total_count if total_count > 0 else 0
        
        metrics_dict = {
            'gap_count': int(gap_count),
            'successful_matches': int(successful_matches),
            'total_tickets_analyzed': int(total_count),
            'gap_ratio': float(round(gap_ratio, 4))
        }
        
        # Plotting Logic
        report_df = pd.DataFrame({
            'Category': ['Content Gap', 'Successful Match'],
            'Count': [gap_count, successful_matches]
        })

        plt.figure(figsize=(7, 4)) 
        sns.barplot(
            x='Category', 
            y='Count', 
            data=report_df, 
            palette=['#FF6347', '#3CB371']
        )
        
        plt.title('Content Gap Analysis (Log History)', fontsize=14)
        plt.ylabel('Number of Tickets', fontsize=12)
        plt.xlabel(f'Total Tickets Analyzed: {total_count} (Gap Ratio: {gap_ratio*100:.1f}%)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(REPORT_IMAGE_PATH)
        plt.close()
        
        return True, REPORT_IMAGE_PATH, metrics_dict

    except Exception as e:
        return False, None, {"message": f"Error generating report: {e}"}


# Define the column structure and weights for dynamic rendering (Unchanged)
COLUMNS_MAP = {
    'ID': {'key': 'index', 'weight': 0.7, 'label': 'ID'},
    'Ticket Snippet': {'key': 'Ticket Snippet', 'weight': 2.0, 'label': 'Ticket Snippet'},
    'Severity': {'key': 'Severity_Label', 'weight': 1.2, 'label': 'Severity'},
    'Assigned Team': {'key': 'Assigned_Team_Label', 'weight': 1.2, 'label': 'Assigned Team'},
    'Top Suggestion': {'key': 'Top Suggestion', 'weight': 1.8, 'label': 'Top Suggestion'},
    'Solution/Action': {'key': 'Action', 'weight': 2.5, 'label': 'Top Solution'}
}
ALL_COLUMN_TITLES = list(COLUMNS_MAP.keys())


def plot_issue_distribution(results_df):
    """Generates a bar plot for the distribution of predicted Issues using Matplotlib."""
    
    if 'Issue_Label' not in results_df.columns:
        st.warning("Issue prediction data not available in results.")
        return

    issue_counts = results_df['Issue_Label'].value_counts().reset_index()
    issue_counts.columns = ['Issue Type', 'Count']
    
    if issue_counts.empty:
        st.warning("No issue data to plot.")
        return

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x='Count', 
        y='Issue Type', 
        data=issue_counts, 
        palette='magma'
    )
    plt.title('Distribution of Predicted Issue Types (Counts)', fontsize=14)
    plt.xlabel('Number of Tickets', fontsize=12)
    plt.ylabel('Predicted Issue Type', fontsize=12)
    plt.tight_layout()
    
    st.pyplot(plt)
    plt.close()


# NEW FUNCTION: Fetches Slack activity using caching
@st.cache_data(ttl=5) 
def fetch_slack_activity():
    """Fetches latest Slack messages from the Flask API, caching the result for 5 seconds."""
    try:
        response = requests.get(SLACK_API_URL)
        response.raise_for_status()
        data = response.json()
        return data.get('messages', []) 
    except requests.exceptions.ConnectionError:
        # Fail silently if backend is down, keeping the rest of the GUI working
        return []
    except Exception:
        # For other unexpected API errors
        return []


# UPDATED FUNCTION: Notification Bell Panel now displays both App Events and Slack Messages
def display_notification_panel(slack_messages_external):
    """
    Displays the application events and real-time Slack chat activity.
    The real-time messages are passed in externally.
    """
    # Calculate count from application events + new chat messages
    app_notif_count = len(st.session_state['notifications'])
    chat_message_count = len(slack_messages_external)
    
    total_notif_count = app_notif_count + chat_message_count

    # Use a badge style for the popover label
    label = f"🔔 Notifications ({total_notif_count})" if total_notif_count > 0 else "🔔 Notifications"

    # Use a popover to create the notification panel effect
    with st.popover(label, use_container_width=True):
        st.subheader("Activity Log")

        # --- 1. Real-Time Slack Chat Messages (New Feature) ---
        st.markdown("##### 💬 New Slack Channel Activity (Real-Time)")
        if slack_messages_external:
            # Display Slack messages (Newest first)
            for msg in slack_messages_external:
                # Assuming 'user', 'time', and 'text' keys from app.py are present
                st.markdown(f"<small>*{msg['time']}* **{msg.get('user', 'App')}**: {msg['text'][:80]}...</small>", unsafe_allow_html=True)
            st.info(f"Showing {chat_message_count} latest messages. Updates every 5 seconds.")
        else:
            st.info("No recent chat messages on Slack (Check `app.py` status).")
            
        st.divider() # Separate real-time chat from application events

        # --- 2. AI Application Events (Original Logic) ---
        st.markdown("##### ⚙️ AI Application Events")
        if st.session_state['notifications']:
            # Display application notifications in reverse chronological order
            for i, notif in enumerate(st.session_state['notifications'][::-1]):
                st.markdown(f"**{notif['icon']} {notif['title']}**")
                st.markdown(f"<small>{notif['message']}</small>", unsafe_allow_html=True)
                
                # Add a separator between notifications
                if i < app_notif_count - 1:
                    st.divider()
            
            # Button to clear the log
            # The label is updated to clear only app events
            if st.button("Clear App Events", type="primary", use_container_width=True):
                 st.session_state.notifications = []
                 st.rerun()
        else:
            st.info("No recent AI application events.")


# Main Application Logic

def main():
    """Builds the Streamlit GUI."""
    
    # 1. Page Configuration
    st.set_page_config(layout="wide")

    # NEW: Fetch Slack messages here (will update every 5 seconds due to cache TTL)
    slack_messages = fetch_slack_activity() 
    
    # --- 2. Sidebar Configuration ---
    # The filters (selected_severities, final_columns_to_display, show_report) are set here
    with st.sidebar:
        st.title("Display Settings")
        
        # Column Visibility Control
        st.markdown("---")
        st.subheader("Column Visibility")
        
        configurable_columns = ['Ticket Snippet', 'Severity', 'Assigned Team', 'Top Suggestion']
        
        visible_columns_titles = []
        for col_title in configurable_columns:
            is_visible = st.checkbox(col_title, value=True, key=f"col_check_{col_title}")
            if is_visible:
                visible_columns_titles.append(col_title)

        final_columns_to_display = ['ID'] + visible_columns_titles + ['Solution/Action']

        # Row Filtering Control (Severity)
        st.markdown("---")
        st.subheader("Row Filters")
        DEFAULT_SEVERITY_OPTIONS = ['Critical', 'High', 'Medium', 'Low', 'N/A']
        selected_severities = st.multiselect(
            "Filter by Predicted Severity",
            options=DEFAULT_SEVERITY_OPTIONS,
            default=DEFAULT_SEVERITY_OPTIONS
        )
        
        # Report Toggle
        st.markdown("---")
        show_report = st.checkbox(
            "Show Content Gap Analysis?", 
            value=False,
            help="Check this box to generate and display the historical gap report after analysis."
        )


    # --- 3. Main Content Area: Header and Input ---
    
    st.title("AI-Powered Support Ticket Classifier")
    
    # --- UPDATED: Notification Bell and Analytics Toggle Buttons ---
    col_title, col_notif, col_analytics = st.columns([7, 1.5, 1.5])
    
    col_title.markdown("Instantly predict ticket **Severity** and **Assigned Team**, and retrieve relevant Knowledge Base articles.")
    
    # 🔔 Notification Panel in col_notif (UPDATED: Pass real-time slack messages)
    with col_notif:
        display_notification_panel(slack_messages)
    
    # 📊 Analytics Toggle in col_analytics
    with col_analytics:
        if st.button("📊 Analytics", type="secondary", use_container_width=True):
            st.session_state.show_analytics = not st.session_state.show_analytics
            # Trigger a rerun to display the analytics section immediately
            st.rerun() 
            
    st.divider()

    st.subheader("Simulate Incoming Ticket")
    
    # Input Area 
    uploaded_file = st.file_uploader(
        "Upload a Ticket File (.txt for bulk, .csv for bulk)",
        type=['txt', 'csv'],
        help="Upload a plain text file (separate tickets with three empty lines) or a CSV with multiple rows of ticket data."
    )

    ticket_text = st.text_area(
        "OR Paste Single Ticket Description Here", 
        height=150,
        placeholder="Type a single customer's issue here..."
    )

    tickets_to_process = []
    
    if uploaded_file is not None:
        tickets_to_process = extract_text_from_upload(uploaded_file)
    elif ticket_text:
        tickets_to_process = [{'text': ticket_text, 'source': 'Manual Input'}]

    if st.button("Get AI Suggestions & Priority", type="primary"):
        if not tickets_to_process:
            st.warning("Please enter or upload a ticket description to begin.")
            # Ensure the function exits here on failure
            return

        all_results = []
        slack_sent_success = True # Flag for overall Slack status
        
        # BULK PROCESSING LOOP 
        with st.spinner(f'Processing {len(tickets_to_process)} tickets...'):
            for i, ticket in enumerate(tickets_to_process):
                # UPDATED: Capture 5 values, including slack_status
                suggestions, severity_info, issue_info, team_info, slack_status = get_predictions(ticket['text'])
                
                # Check for API failure
                if suggestions is None: 
                    return
                
                # NEW: Update the overall slack status flag
                if not slack_status:
                    slack_sent_success = False 

                top_suggestion = suggestions[0] if suggestions else None
                
                # Format prediction scores
                sev_display = display_prediction_score(severity_info)
                team_display = display_prediction_score(team_info)
                issue_label = display_prediction_score(issue_info)['label']

                # Store ALL necessary details
                all_results.append({
                    'index': i,
                    'Source': ticket['source'],
                    'Ticket Snippet': ticket['text'][:80] + '...',
                    'Severity_Label': sev_display['label'], 
                    'Severity_Display': f"**{sev_display['label']}** ({sev_display['score_text']})",
                    'Assigned_Team_Label': team_display['label'], 
                    'Assigned_Team_Display': f"**{team_display['label']}** ({team_display['score_text']})",
                    'Issue_Label': issue_label, # New column for issue type
                    'Top Suggestion': top_suggestion['title'] if top_suggestion else "Content Gap Detected",
                    'Suggestions': suggestions, 
                    'Severity_Raw': sev_display['label'] 
                })
        
        # --- Notification Panel Logic (REPLACED st.toast) ---
         # --- Notification Panel Logic (Improved Slack Status Detection) ---
        total_tickets = len(tickets_to_process)

        # Count how many Slack messages actually succeeded
        slack_success_count = sum(1 for t in all_results if t.get('Suggestions'))

        if slack_success_count == total_tickets:
            title = f"Slack Success for {total_tickets} Tickets"
            message = "All automatic Slack notifications were sent successfully."
            icon = '✅'
        elif slack_success_count > 0:
            title = f"⚠️ Partial Slack Success"
            message = f"Slack sent {slack_success_count}/{total_tickets} notifications successfully."
            icon = '⚠️'
        else:
            title = f"❌ Slack Failure for {total_tickets} Tickets"
            message = "No Slack notifications were confirmed. Check `app.py` logs for details."
            icon = '❌'

        st.session_state['notifications'].append({
            'title': title,
            'message': message,
            'icon': icon
        })

        st.success(f'Analysis of {len(tickets_to_process)} tickets complete. Check the **🔔 Notifications** panel for Slack status.')
        st.session_state['all_results_df'] = pd.DataFrame(all_results)
        st.rerun()

        # Provide immediate feedback on completion
        st.success(f'Analysis of {len(tickets_to_process)} tickets complete. Check the **🔔 Notifications** panel for Slack status.')
        
        # Store results in session state for analytics section
        st.session_state['all_results_df'] = pd.DataFrame(all_results)
        
        # CRITICAL FIX: Force a rerun to update the page, ensuring the notification badge 
        # is correctly updated with the new count in session state.
        st.rerun()


    # --- 4. Display Results (MOVED OUTSIDE THE BUTTON BLOCK) ---
    # The display logic must be outside the button's IF block to persist on subsequent interactions.
    if not st.session_state['all_results_df'].empty:
        # Retrieve the analysis results from session state
        results_df = st.session_state['all_results_df']
        all_results = results_df.to_dict('records')

        st.subheader(f"Analysis Complete: {len(all_results)} Tickets Processed")
        st.markdown("### Interactive Analysis Summary")
        
        # Apply Row Filtering using the DataFrame
        filtered_df = results_df[results_df['Severity_Raw'].isin(selected_severities)]
        filtered_results = filtered_df.to_dict('records')
        
        st.info(f"Displaying **{len(filtered_results)}** out of **{len(all_results)}** results based on filters.")
        
        if not filtered_results:
            st.warning("No results match the current filters.")
        else:
            # 4.1 Create Header Row
            column_weights = [COLUMNS_MAP[col]['weight'] for col in final_columns_to_display]
            
            header_cols = st.columns(column_weights)
            for i, col_title in enumerate(final_columns_to_display):
                header_cols[i].markdown(f"**{col_title}**")
            st.markdown("---")


            # 4.2 Row-by-Row Interactive Display
            for result in filtered_results:
                cols = st.columns(column_weights)
                top_suggestion = result['Suggestions'][0] if result['Suggestions'] else None
                col_index = 0

                for col_title in final_columns_to_display:
                    current_col = cols[col_index]
                    col_index += 1
                    
                    if col_title == 'ID':
                        current_col.markdown(f"**#{result['index'] + 1}**")
                    
                    elif col_title == 'Ticket Snippet':
                        current_col.markdown(result['Ticket Snippet'])
                        
                    elif col_title == 'Severity':
                        current_col.markdown(result['Severity_Display'])
                        
                    elif col_title == 'Assigned Team':
                        current_col.markdown(result['Assigned_Team_Display'])
                        
                    elif col_title == 'Top Suggestion':
                        current_col.markdown(result['Top Suggestion'])

                    elif col_title == 'Solution/Action':
                        if top_suggestion:
                            with current_col:
                                with st.expander("▶ **Show Recommended Solution**", expanded=False):
                                    st.markdown(f"**Top Match:** {top_suggestion['title']}")
                                    st.markdown(f"**Confidence:** {top_suggestion['similarity_score'] * 100:.2f}%")
                                    st.markdown("---")
                                    
                                    resolution_text = top_suggestion.get('resolution_steps', 'Resolution steps unavailable.')
                                    steps = resolution_text.split('; ')
                                    st.markdown("**Resolution Steps:**")
                                    for step in steps:
                                        st.markdown(f"**-** {step.strip()}")
                        else:
                            current_col.error(" **Content Gap Detected**")
                
                st.markdown("---")

    
    # --- 5. Conditional Analytics Section ---
    if st.session_state.show_analytics:
        st.divider()
        st.subheader("📊 Model Prediction Analytics")
        st.markdown("Analyze the distribution of predictions from the last run.")
        
        if not st.session_state['all_results_df'].empty:
            results_df = st.session_state['all_results_df']
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("#### Predicted Issue Types")
                # Visualization for Issue distribution (using Matplotlib/Seaborn)
                plot_issue_distribution(results_df)

            with col_right:
                st.markdown("#### Predicted Severity Distribution")
                severity_counts = results_df['Severity_Label'].value_counts().reset_index()
                severity_counts.columns = ['Severity', 'Count']
                
                if not severity_counts.empty:
                    # Visualization for Severity distribution (using Plotly)
                    fig = px.pie(
                        severity_counts, 
                        values='Count', 
                        names='Severity', 
                        title='Predicted Severity Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No severity data to plot.")

        else:
            st.warning("No bulk analysis results available to display analytics. Run an analysis first.")


    # --- 6. Content Gap Reporting Hub (Original Logic) ---
    if show_report:
        st.divider()
        st.subheader("Real-Time Content Gap Analysis")
        
        success, report_output_image, report_output_metrics = generate_gap_report()
        
        if success:
            st.markdown("### Metrics (Vector Analysis)")
            
            col_m1, col_m2, col_m3, col_m4 = st.columns([1, 1, 1, 1])
            
            col_m1.metric(
                label="Content Gap Ratio", 
                value=f"{report_output_metrics['gap_ratio'] * 100:.2f}%",
                help="Percentage of tickets with low/no matching KB articles."
            )
            col_m2.metric("Total Tickets Analyzed", report_output_metrics['total_tickets_analyzed'])
            col_m3.metric("Tickets Flagged as Gap", report_output_metrics['gap_count'])
            col_m4.metric("Successful KB Matches", report_output_metrics['successful_matches'])
            
            st.markdown("### Chart Visualization")
            
            col_left, col_center, col_right = st.columns([2, 3, 2])
            
            with col_center:
                st.image(
                    report_output_image, 
                    caption=f"Report updated at {pd.Timestamp.now().strftime('%H:%M:%S')}",
                )
        else:
            st.warning(f"Could not generate report: {report_output_metrics.get('message', report_output_image)}")


if __name__ == "__main__":
    main()