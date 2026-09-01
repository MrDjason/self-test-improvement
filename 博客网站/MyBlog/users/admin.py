from django.contrib import admin

from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from .models import UserProfile, EmailVerifyRecord # 引入定义的模型

# 必须先通过unregister将User取消注册
admin.site.unregister(User)

# 定义关联对象的样式，StackedInline为纵向排列每一行，TabularInline为并排排列
class UserProfileInline(admin.StackedInline):
    model = UserProfile # 关联的模型

# 关联字段在User之内编辑，关联进来
class UserProfileAdmin(UserAdmin):
    inlines = [UserProfileInline]

# 重新注册User
admin.site.register(User, UserProfileAdmin)

@admin.register(EmailVerifyRecord)
class Admin(admin.ModelAdmin):
    '''Admin View for'''
    list_display = ('code',)