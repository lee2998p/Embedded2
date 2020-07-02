import smtplib
from email.mime.text import MIMEtext
from email.mime.multipart import MIMEmultipart
from email.mime.base import MIMEBase
from crontabs import Cron, Tab
from datetime import datetime

def send_email(sender_addr, receiver_addr, msg, body, filename):
  try:
    msg.attach(MIMEText(body, 'plain'))
    msg.attach(attach_file(filename))
    text = msg.as_string()
    server = smtplib.SMTP('smtp.office365.com', 587)
    server.ehlo()
    server.starttls()         #secure connection
    server.login(email, password)

    server.sendmail(sender_addr, receiver_addr, text)
    server.quit()
    print('Email successfully sent.')
  except:
    print('Email failed to be sent.')
    
def attach_file(filename):
  try:
    attachment = open(filename, 'rb')

    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= "+filename)
    return(part)
    print('File successfully attached.')
  except:
    print('File failed to be attached.')
    
sender_addr = 'Sender Address'
receiver_addr = 'Recipient Address'

msg = MIMEMultipart()
msg['From'] = sender_addr
msg['To'] = receiver_addr
msg['subject'] = 'Cam2 Embedded Vision 2020 Video Data'

body='Hello. These are today's video data.'

filename='video.mp4'

email=" "
password=" "

Cron().schedule(
  Tab(
    name='send_email'
  ).run(
    send_email
  ).every(
    day=1
  ).starting(
    timestamp.hour=8
  )
).go()
