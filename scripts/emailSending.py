import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Gmail SMTP server setup
smtp_server = 'smtp.gmail.com'
smtp_port = 587  # TLS port

# Email content
def sendMail(body, to):
    msg = MIMEMultipart()  # Use MIMEMultipart for emails with both text and HTML parts
    sender = 'venkatsai4431@gmail.com'
    recipients = to
    msg['Subject'] = "Subject"
    msg['From'] = sender
    msg['To'] = ", ".join(recipients)

    # Attach HTML body content
    msg.attach(MIMEText(body, 'html'))  # Send the body as HTML

    # Start SMTP session
    s = smtplib.SMTP(smtp_server, smtp_port)

    # Start TLS for security
    s.starttls()

    # Login to Gmail
    s.login(sender, 'mgum iade aemg kdjk')  # Replace with your Gmail App Password

    # Send the email
    try:
        s.sendmail(sender, recipients, msg.as_string())
        s.quit()
        return "Email Sent", True
    except Exception as e:
        return f"Email is not sent, error: {e}", False
