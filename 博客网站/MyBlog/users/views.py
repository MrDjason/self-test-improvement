from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth import authenticate, login
from .forms import LoginForm, RegisterForm

from django.contrib.auth.backends import ModelBackend
from django.contrib.auth.models import User
from django.db.models import Q

from django.db import models

from .models import EmailVeriftRecord

class MyBackend(ModelBackend):
    '''邮箱登录注册'''
    def authenticate(self, request, username=None, password=None, **kwargs):
        try:
            user = User.objects.get(Q(username=username)|Q(email=username))
            if user.check_password(password):
                return user
        except Exception as e:
            return None

def active_user(request, active_code):
    all_records = EmailVeriftRecord.objects.filter(code=active_code)
    if all_records:
        for record in all_records:
            email = recod.email
            user = User.objects.get(email=email)
            user.is_staff = True
            user.save()
    else:
        return HttpResponse('链接有误')
    return redirect('users:login')

def login_view(request):
    if request.method != 'POST':
        form = LoginForm()
    else:
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('/admin')
            else:
                return HttpResponse('账号或密码错误！')
    return render(request, 'users/login.html', {'form':form})

def register(request):
    # 注册视图
    if request.method != 'POST':
        form = RegisterForm()
    else:
        form = RegisterForm(request.POST)
        if form.is_valid():
            new_user = form.save(commit=False)
            new_user.set_password(form.cleaned_data.get('password'))
            new_user.username = form.cleaned_data.get('email') # 注册添加邮箱功能
            new_user.save()
            return HttpResponse('注册成功')
    return render(request, 'users/register.html', {'form':form})