import smtplib
from email.mime.text import MIMEText


def send_mail(to_emails: list, content: str, subject="", server="smtp.qq.com", from_email="xxxx@qq.com", password=""):
    message = MIMEText(content, "plain", "utf-8")  # 内容, 格式, 编码
    message["From"] = from_email
    message["To"] = ",".join(to_emails)
    message["Subject"] = subject

    try:
        server = smtplib.SMTP_SSL("smtp.qq.com", 465)
        server.login(from_email, password)
        server.sendmail(from_email, from_email, message.as_string())
        server.quit()
        print("successfully sent the mail.")
    except smtplib.SMTPException as e:
        print(e)


content = f"Model Training Done."
send_mail(["xxx@qq.com"], content=content, subject=content)
