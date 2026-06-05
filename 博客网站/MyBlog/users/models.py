from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    '''
    owner = models.OneToOneField(User,
    on_delete=models.CASCADE, # 删除则关联数据全部删除
    verbose_name = '用户' # 定义名称
    )
    '''
    USER_GENDER_TYPE = (
        ('male', '男'),
        ('female', '女')
    )
    owner = models.OneToOneField(User, on_delete=models.CASCADE, verbose_name = '用户')
    nike_name = models.CharField('昵称', max_length=50, blank=True, default='')
    birthday = models.DateField('生日', null=True, blank=True)
    gender = models.CharField('性别', max_length=6, choices=USER_GENDER_TYPE)
    address = models.CharField('地址', max_length=100, blank=True, default='')
    image = models.ImageField(upload_to='images/%Y/%m', default='images/default.png', max_length=100, verbose_name='用户头像')

    class Meta:
        verbose_name = '用户数据'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.owner.username