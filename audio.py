import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd
from scipy.io.wavfile import write

frequency_sampling, audio_signal = wavfile.read("D:/amthanh.wav")
print('\nSignal shape:', audio_signal.shape)
print('Signal Datatype:', audio_signal.dtype)

power = []
power1 = []
signal_vcc = []
audio_signal = audio_signal * 5  # / 500000
for i in range(len(audio_signal)):
    if audio_signal[i] > 0:
        signal_vcc.append(audio_signal[i])
    else:
        signal_vcc.append(0)

length_signal = len(audio_signal)
half_length = np.ceil((length_signal + 1) / 2.0).astype(np.int)
signal_frequency = np.fft.fft(audio_signal)
signal_frequency = abs(signal_frequency[0:half_length])
for i in range(len(signal_frequency)):
    power.append(int(math.sqrt(
        signal_frequency[i].real * signal_frequency[i].real + signal_frequency[i].imag * signal_frequency[i].imag)))
signal_power = 10 * np.log10(signal_frequency)
x_axis = np.arange(0, len(signal_frequency), 1)  # * (frequency_sampling / length_signal) / 1000.0
for j in range(len(signal_frequency)):
    if j > 500:
        power1.append(signal_frequency[j])
len_speach = len(signal_vcc)
value = []
value1 = []
value2 = []
value3 = []
value4 = []
m = 0
for i in range(len(signal_vcc)):
    if i < 1:
        value.append(0)
    if i >= 1 and i < len(signal_vcc) - 1:
        if signal_vcc[i] > (signal_vcc[i - 1] * 2) or signal_vcc[i] > (signal_vcc[i + 1] * 2):
            for j in range(i - m - 1):
                value.append(0)
            value.append(int(signal_vcc[i]))
            m = i

print('len_signal:', len_speach)
print('lenvalue:', len(value))
# m=0
# for i in range(len(value)):
#	if i>2:
#		if value[i]>0:
#			if value[i]>2*m or value[i]*2<m:
#				print(1)
#				m=value[i]
#			else:
#				m=value[i]
#				value[i]=0

k = value[0]
ind = 0
for i in range(len(value)):
    if i > 0:
        if value[i] > 0:
            value[i] = value[ind] = (value[i] + value[ind]) / 2
            ind = i
k = value[0]
ind = 0
for i in range(len(value)):
    if i > 0:
        if value[i] > 0:
            value[i] = value[ind] = (value[i] + value[ind]) / 2
            ind = i
ind = 0
for i in range(len(value)):
    if value[i] > 20:
        if value[i] > (1.3 * value[ind]):
            for j in range(i - ind):
                value1.append(0)
            value1.append(int(value[i]))
            ind = i
        if value[i] * 1.3 < value[ind]:
            for j in range(i - ind):
                value1.append(0)
            value1.append(int(value[ind]))
            ind = i
ind = 0
for i in range(len(value1)):
    if value1[i] > 20:
        if value1[i] > (2 * value1[ind]):
            for j in range(i - ind - 1):
                value2.append(0)
            value2.append(int(value1[i]))
            ind = i
        if value1[i] * 2 < value1[ind]:
            for j in range(i - ind - 1):
                value2.append(0)
            value2.append(int(value1[ind]))
            ind = i
list_kc = []  # Chứa thông tin khoảng cách thời gian giữa các từ trong câu nói (của tất cả âm nhiễu nữa)
list_ind = []  # chứa thông tin thời điểm thời gian bắt đầu của các từ trong câu nói
kc = 0
for i in range(len(value1)):
    if value1[i] > 300:
        list_kc.append(i - kc)
        list_ind.append(i)
        kc = i
    if i == len(value1) - 1:
        list_kc.append(i - kc)
        list_ind.append(i)

t = 0
kq = []  # Lưu các giá trị khoảng cách giữa các từ
ind_kq = []
print(list_kc)
print(list_ind)
for i in range(len(list_kc)):
    kk = 1
    if i > 0:
        if list_kc[i - 1] > t and list_kc[i - 1] > 1000:
            if len(list_kc) - i > 12 and i > 10:  # đặt phạm vi giới hạn đầu và cuối để căn chỉnh
                for l in range(10):  # điều kiện nó phải lớn hơn mấy thằng đứng trước nó
                    if list_kc[i - 1] <= list_kc[i + l]:
                        kk = 0
                for s in range(8):  # điều kiện nó phải lớn hơn mấy thằng đứung sau nó
                    if list_kc[i - 1] < list_kc[i - 2 - s]:
                        kk = 0
                if kk == 1:
                    # print(list_kc[i-1])
                    kq.append(list_kc[i - 1])
                    ind_kq.append(i - 1)
            else:
                if list_kc[i - 1] > list_kc[i]:
                    # print(list_kc[i-1])
                    kq.append(list_kc[i - 1])
                    ind_kq.append(i - 1)
        if i == len(list_kc) - 1 and list_kc[len(list_kc) - 1] > list_kc[len(list_kc) - 2]:
            kq.append(list_kc[len(list_kc) - 1])

        t = list_kc[i - 1]

# for m in range(60):
#	for i in range(len(value)):
#		if i>1:
#			value[i-1]=value[i]=(value[i-1]+value[i])/2


print(kq)
print(len_speach)
dai = 0
vitri = []
dodai = []
for i in range(len(ind_kq)):
    dai += (kq[len(ind_kq) - i])
    print('dai:', dai)
    print('Thời gian bắt đầu: ', list_ind[ind_kq[len(ind_kq) - 1 - i]])
    vitri.append(list_ind[ind_kq[len(ind_kq) - 1 - i]])
    print('ĐỘ dài của từ: ', len_speach - dai - list_ind[ind_kq[len(ind_kq) - 1 - i]])
    dodai.append(len_speach - dai - list_ind[ind_kq[len(ind_kq) - 1 - i]])
    print('khoangcach:', list_ind[ind_kq[len(ind_kq) - 1 - i]])
    dai += len_speach - dai - list_ind[ind_kq[len(ind_kq) - 1 - i]]
list_cut_signal = []  # chuỗi chứa các đoạn âm của các từ đã cắt
for i in range(len(vitri)):
    list_cut_signal.append(audio_signal[vitri[i] - 6000:vitri[i] + dodai[i]] + 6000)

samplerate = 44100;
fs = 100

amplitude = np.iinfo(np.int16).max
for i in range(len(list_cut_signal)):  # lưu từng từ vào folder
    write("example" + str(i) + ".wav", frequency_sampling, list_cut_signal[i])

plt.figure(1)
plt.plot(list_cut_signal[2], color='blue')
plt.xlabel('Time (2e-5 seconds)')
plt.ylabel('Amplitude')
# đo thi 1


plt.figure(1)
plt.plot(kq, color='blue')
plt.xlabel('Time (2e-5 seconds)')
plt.ylabel('Amplitude')
plt.figure(2)
plt.plot(ind_kq, color='blue')
plt.xlabel('Time (2e-5 seconds)')
plt.ylabel('Amplitude')

# đô thị 2

plt.figure("khopangcach")
plt.plot(list_kc, color='blue')
plt.xlabel('Time (2e-5 seconds)')
plt.ylabel('Amplitude')

plt.figure('list_ind')
plt.plot(list_ind, color='blue')
plt.xlabel('Time (2e-5 seconds)')
plt.ylabel('Amplitude')

plt.figure('signal')
plt.plot(audio_signal, color='blue')
plt.xlabel('Time (2e-5 seconds)')
plt.ylabel('Amplitude')

# đồ thị 3

plt.figure('value1')
plt.plot(value1, color='blue')
plt.xlabel('Time (2e-5 seconds)')
plt.ylabel('Amplitude')

plt.show()


# s = {'nang luong':power1,'tan so':list(i for i in range(len_power1))}
# df = pd.DataFrame(s)
# df.to_csv("signal_data_ff_6_ts.csv")
