import pandas as pd
import gdown
import smtplib
from email.mime.text import MIMEText
import os

# -------- STEP 1: DOWNLOAD DATA FROM GOOGLE DRIVE --------
file_id = "1QtGmqlvamOfBQK7hB2BWzffNrbqryvjU"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, "data.csv", quiet=True)

# -------- STEP 2: LOAD DATA --------
df = pd.read_csv("data.csv")

# -------- STEP 3: FILTER HIGH RISK TRANSACTIONS --------
high_risk = df[df["isFraud"] == 1].head(10)
if high_risk.empty:
    print("No high-risk transactions found")
    exit()

# -------- STEP 4: CREATE HTML EMAIL --------
table_html = high_risk[['amount', 'type']].to_html(index=False)
html_content = f"""
<h2 style="color:#ef4444;">🚨 Fraud Alert Report</h2>
<p>The following high-risk transactions were detected:</p>
{table_html}
<hr>
<p>
🔗 <b>View Full Dashboard:</b><br>
<a href="https://w1954810finalyearproject-prrtwxqdvhcbabbmyyxuwk.streamlit.app/" target="_blank">
Open Fraud Detection Dashboard
</a>
</p>
<p style="color:gray; font-size:12px;">
Generated automatically by Fraud Detection System
</p>
"""

# -------- STEP 5: SET EMAIL --------
msg = MIMEText(html_content, "html")
msg["Subject"] = "🚨 High Risk Fraud Alert"
msg["From"] = os.environ["EMAIL_SENDER"]
receivers = os.environ["EMAIL_RECEIVER"].split(",")
msg["To"] = ", ".join(receivers)

# -------- STEP 6: SEND EMAIL --------
with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(
        os.environ["EMAIL_SENDER"],
        os.environ["EMAIL_PASSWORD"]
    )
    server.sendmail(
        os.environ["EMAIL_SENDER"],
        receivers,
        msg.as_string()
    )
print("Email sent successfully!")
