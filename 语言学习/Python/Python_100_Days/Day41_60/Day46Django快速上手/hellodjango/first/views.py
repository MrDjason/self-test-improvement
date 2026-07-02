from django.shortcuts import render

from django.http import HttpResponse
from random import sample

def show_index(request):
    fruits = [
        'Apple', 'Orange', 'Pitaya', 'Durian', 'Waxberry', 'Blueberry',
        'Grape', 'Peach', 'Pear', 'Banana', 'Watermelon', 'Mango'
    ]
    selected_fruits = sample(fruits, 3) # sample(序列, k) 从指定序列无序随机选取k各元素返回一个新列表
    return render(request, 'index.html', {'fruits': selected_fruits})
    # render(请求对象request,需要渲染模板页名字,渲染到页面上数据)