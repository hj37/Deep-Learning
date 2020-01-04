
# coding: utf-8

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import json
import pandas as pd 
from konlpy.tag import Twitter
from itertools import chain
from collections import defaultdict
import re
import unicodedata
import string

n_letters = 128

# 각 클래스 별 단어 목록인 answer_lines 사전 생성
answer_lines = {}
all_answers = []
fw = open("test.txt",'w',encoding = 'utf-8')

with open('Tr.json',encoding="iso-8859-1") as f:
    json_data = json.load(f,encoding = "utf-8")    
    for i in range(len(json_data)):
        for j in json_data[i]['sentence']:
            all_answers.append(json_data[i]['answer'])
            all_answers = list(set(all_answers))  #중복요소 제거해서 answer 모든 정답 담기 
            json_data[i]['sentence'] = j.encode('iso-8859-1').decode('euc-kr', 'ignore')
            

Tr = pd.DataFrame(json_data)
Tr_1_answer = Tr.loc[(Tr['answer'] == '001')]
print(Tr_1_answer)
Tr_2_answer = Tr.loc[(Tr['answer'] == '002')]
Tr_3_answer = Tr.loc[(Tr['answer'] == '003')]
Tr_4_answer = Tr.loc[(Tr['answer'] == '004')]
Tr_5_answer = Tr.loc[(Tr['answer'] == '005')]
#print(Tr_1_answer)
Tr_1_answer_co = ' '.join(re.findall(r'([ㄱ-힣]*/NN|[ㄱ-힣]*/VV|[a-z]*/SL|[0-9]*/SN)+',str(Tr_1_answer['sentence'])))
Tr_2_answer_co = ' '.join(re.findall(r'([ㄱ-힣]*/NN|[ㄱ-힣]*/VV|[a-z]*/SL|[0-9]*/SN)+',str(Tr_2_answer['sentence'])))
Tr_3_answer_co = ' '.join(re.findall(r'([ㄱ-힣]*/NN|[ㄱ-힣]*/VV|[a-z]*/SL|[0-9]*/SN)+',str(Tr_3_answer['sentence'])))
Tr_4_answer_co = ' '.join(re.findall(r'([ㄱ-힣]*/NN|[ㄱ-힣]*/VV|[a-z]*/SL|[0-9]*/SN)+',str(Tr_4_answer['sentence'])))
Tr_5_answer_co = ' '.join(re.findall(r'([ㄱ-힣]*/NN|[ㄱ-힣]*/VV|[a-z]*/SL|[0-9]*/SN)+',str(Tr_5_answer['sentence'])))

print(Tr_1_answer_co)
Tr_1_nouns = re.sub('[A-Z/]','',Tr_1_answer_co)
Tr_2_nouns = re.sub('[A-Z/]','',Tr_2_answer_co)
Tr_3_nouns = re.sub('[A-Z/]','',Tr_3_answer_co)
Tr_4_nouns = re.sub('[A-Z/]','',Tr_4_answer_co)
Tr_5_nouns = re.sub('[A-Z/]','',Tr_5_answer_co)

answer_lines['001']= Tr_1_nouns.split()
answer_lines['002'] = Tr_2_nouns.split()
answer_lines['003'] = Tr_3_nouns.split()
answer_lines['004'] = Tr_4_nouns.split()
answer_lines['005'] = Tr_5_nouns.split()
print(answer_lines['001'])
                                                      
n_answers = len(all_answers)


# In[4]:


print(answer_lines)


# In[28]:


import torch

# all_letters 로 문자의 주소 찾기, 예시 "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# 검증을 위해서 한 문자를 <1 x n_letters> Tensor로 변환  #0NE-HOT벡터로 만드는 과정 
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# 한 줄(이름)을  <line_length x 1 x n_letters>,
# 또는 문자 벡터의 Array로 변경
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('가'))

print(lineToTensor('가나').size())


# In[29]:


import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_answers)


# In[30]:


input = letterToTensor('가')
hidden =torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)
print(output)


# In[31]:


input = lineToTensor('음식점')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)


# In[32]:


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    answer_i = top_i[0].item()
    return all_answers[answer_i], answer_i

print(categoryFromOutput(output))


# In[33]:


import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    answer = randomChoice(all_answers)
    line = randomChoice(answer_lines[answer])
    answer_tensor = torch.tensor([all_answers.index(answer)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return answer, line, answer_tensor, line_tensor

for i in range(1000):
    answer, line, answer_tensor, line_tensor = randomTrainingExample()
    print('answer =', answer, '/ line =', line)


# In[34]:


criterion = nn.NLLLoss()


# In[35]:


learning_rate = 0.005 # 이것을 너무 높게 설정하면 폭발할 수 있고 너무 낮으면 학습이 되지 않을 수 있습니다.

def train(answer_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, answer_tensor)
    loss.backward()

    # learning rate를 곱한 파리미터의 경사도를 파리미터 값에 더합니다.
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


# In[36]:


import time
import math

n_iters = 100000
print_every = 5000
plot_every = 5000



# 도식화를 위한 소실 추적
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    answer, line, answer_tensor, line_tensor = randomTrainingExample()
    output, loss = train(answer_tensor, line_tensor)
    current_loss += loss

    # iter 숫자, 손실, 이름, 추측 출력
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == answer else '✗ (%s)' % answer
        print('iterators : %d 퍼센테이지: %d%% 걸린시간: (%s) loss: %.4f 어구: %s  예측 클래스: %s 정답: %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # 현재 평균 손실을 손실 리스트에 추가
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


# In[37]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)


# In[38]:


# 혼란 행렬에서 정확한 추측을 추적
confusion = torch.zeros(n_answers, n_answers)
n_confusion = 10000

# 주어진 라인의 출력 반환
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# 올바르게 추측 된 예시와 기록을 살펴보십시오.
for i in range(n_confusion):
    answer, line, answer_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    answer_i = all_answers.index(answer)
    confusion[answer_i][guess_i] += 1

# 모든 행을 합계로 나눔으로써 정규화하십시오.
for i in range(n_answers):
    confusion[i] = confusion[i] / confusion[i].sum()

# 도식 설정
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# 축 설정
ax.set_xticklabels([''] + all_answers, rotation=90)
ax.set_yticklabels([''] + all_answers)

# 모든 tick에서 강제로 레이블 지정
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()


# In[ ]:


#

