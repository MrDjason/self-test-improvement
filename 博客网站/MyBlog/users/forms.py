from django import forms
from django.contrib.auth.models import User

class LoginForm(forms.Form):
    username = forms.CharField(label='用户名', max_length=32, widget=forms.TextInput(attrs={
        'class' : 'input', 'placeholder' : '用户名/邮箱'
    }))
    password = forms.CharField(label='密码', min_length=6, widget=forms.PasswordInput(attrs={
        'class' : 'input', 'placeholder' : '密码'
    }))

    def clean_password(self):
        username = self.cleaned_data.get('username')
        password = self.cleaned_data.get('password')

        if username  == password:
            raise forms.ValidationError('用户名与密码不能相同')
        return password
    
class RegisterForm(forms.ModelForm):
    email = forms.CharField(label='邮箱',min_length=3, widget=forms.TextInput(attrs={
        'class':'input', 'placeholder':'邮箱'}))
    password = forms.CharField(label='密码', min_length=6, widget=forms.PasswordInput(attrs={
        'class' : 'input', 'placeholder' : '密码'
    }))
    password1 = forms.CharField(label='再次输入密码', min_length=6, widget=forms.PasswordInput(attrs={
        'class' : 'input', 'placeholder' : '再次输入密码'
    }))

    class Meta:
        model = User
        fields = ('email', 'password')

    def clean_email(self):
        email = self.cleaned_data.get('email')
        exists = User.objects.filter(email=email).exists()
        if exists:
            raise forms.ValidationError('用户名已经存在')
        return email
    
    def clean_password1(self):
        data = self.cleaned_data
        password = data['password']
        password1 = data['password1']
        if password != password1:
            raise forms.ValidationError('两次输入的密码不一致')
        return password
    
class ForgetpwdForm(forms.Form):
    '''填写邮箱地址表单'''
    email = forms.EmailField(label='请输入注册邮箱地址', min_length=4, widget=forms.EmailInput(attrs={
        'class':'input', 'placeholder':'用户名/邮箱'
    }))

class ModifyPwdForm(forms.Form):
    '''修改密码表单'''
    password = forms.CharField(label='输入新密码', min_length=6, widget=forms.PasswordInput(attrs={'class':'input', 'placeholder':'输入密码'}))