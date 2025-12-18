from django.shortcuts import render

# Create your views here.
# request
from django.http import HttpRequest
from django.http import HttpResponse

# 我们期望用输入http://127.0.0.1:8000/index/访问视图函数
def index(request):
    # request, template_name, context=None

    # 模拟数据查询
    context = {
        'name':'马上双十一，点击有惊喜'
    }
    return render(request, 'book/index.html', context=context)