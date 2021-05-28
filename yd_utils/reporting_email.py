import smtplib
from datetime import datetime



def send_email(target_mail_address_list, server_name="Default", exception_message="", successfully=False):
    """Sending reporting email.
    Args:
        exception_message (str, optional): [description]. Defaults to "".
        successfully (bool, optional): Defines if the email contains a finishing message or an error message. Defaults to False.
    """
    email_server_ip = '64.233.184.108'
    sender_mail_address = "server.report.001@gmail.com"
    sender_mail_pw = "sokukyuuvcuyhpsg"

    for target_mail_address in target_mail_address_list:
        with smtplib.SMTP(email_server_ip, 587) as smtp:
            smtp.ehlo()  
            smtp.starttls()
            smtp.ehlo()

            smtp.login(sender_mail_address, sender_mail_pw)

            if successfully is False:
                subject = server_name +': Error, time: {}'.format(str(datetime.now()))
                body = 'Error\nTimestamp: ' + str(datetime.now()) + \
                    '\nThe error message is:\n' + str(exception_message)
                msg = f'Subject: {subject}\n\n{body}'
                smtp.sendmail(sender_mail_address, target_mail_address, msg)
                print("Error mail sent")

            elif successfully is True:
                subject = server_name +': Success, time: {}'.format(str(datetime.now()))
                body = 'Success\nTimestamp: ' + str(datetime.now())
                msg = f'Subject: {subject}\n\n{body}'
                smtp.sendmail(sender_mail_address, target_mail_address, msg)
                print("Success mail sent.")

# send_email("test", target_mail_address=target_mail_address, successfully=False, )