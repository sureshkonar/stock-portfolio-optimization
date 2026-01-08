import smtplib
import os
from email.message import EmailMessage
from datetime import datetime

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("EMAIL_TO")

FILE_PATH = "Daily_StockIQ_Insights.xlsx"

msg = EmailMessage()
msg["Subject"] = "📊 Daily StockIQ Insights Report"
msg["From"] = EMAIL_USER
msg["To"] = EMAIL_TO

msg.set_content(f"""
************** This is a auto-generated email **************
************** Please do not reply *************************
                      
Hello,

Your Daily StockIQ Insights Report has been generated successfully.

📅 Date: {datetime.now().strftime('%d %b %Y')}
📎 Attachment: Excel report

Regards,
StockIQ Automation 🚀
""")

with open(FILE_PATH, "rb") as f:
    msg.add_attachment(
        f.read(),
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=FILE_PATH
    )

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(EMAIL_USER, EMAIL_PASS)
    server.send_message(msg)

print("📧 Email sent successfully")
