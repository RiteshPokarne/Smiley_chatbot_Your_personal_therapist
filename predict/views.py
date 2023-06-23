from django.http import response
from django.shortcuts import render,redirect
import pickle
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
# Create your views here.s
import requests
from django.views.decorators.csrf import csrf_exempt
from .predictions import predict_bot
d={}
inp=''
system=''
version=''

def home(request):
    return render(request,"home.html",{inp:inp})

@csrf_exempt
def chatbot(request):
    global d
    print("hi")
    if request.method == 'GET':            
        inp= request.GET.get('inp')
        print(inp)
    predictions=predict_bot(inp)
    d={"inp":inp,"predictions":predictions}
    return render(request,"home.html",{"d":d})
            # return response.JsonResponse(a,safe=False)
# def display(request):

#     return response.JsonResponse(prediction_label,safe=False)

def execu(request):
    return render(request,"message.html",{"inp":inp,"system":system,"version":version})
def exec(request):
    if request.method == 'GET':    
        global inp,system,version        
        inp= request.GET.get('address')
        system=request.GET.get('system')
        version=request.GET.get('version')
        print(inp)
    return redirect(execu)
