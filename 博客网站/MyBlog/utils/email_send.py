from users.models import EmailVerifyRecord
from django.core.mail import send_mail
import random
import string

def random_str(randomlength=8):
    '''生成8位数的随机字符串'''
    chars = string.ascii_letters + string.digits
    strcode = ''.join(random.sample(chars, randomlength))
    return strcode

def send_register_email(email, send_type='register'):
    email_record = EmailVerifyRecord()
    code = random_str()
    email_record.code = code
    email_record.email = email
    email_record.send_type = send_type
    email_record.save()

    if send_type == 'register':
        email_title = '博客的注册激活链接'
        email_body = f'请点击以下链接激活账号：http://127.0.0.1:8000/users/active/{code}/'
        send_status = send_mail(email_title, email_body, '13218798893@163.com', [email])
        if send_status:
            pass

    elif send_type == 'forget':
        email_title = '找回密码链接'
        email_body = f'请点击以下链接修改密码：http://127.0.0.1:8000/users/forget_pwd_url/{code}/'

        send_status = send_mail(email_title, email_body, '13218798893@163.com', [email])
        if send_status:
            pass