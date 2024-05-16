# perceptron concept
perceptron คือ โครงข่ายประสาทเทียม เป็นองค์ประกอบสำคัญที่อยู่ใน Neural Network
ใช้ในการจำลองการทำงานของ neuron cell

# perceptron formula
## O = act(∑wx+b)
>> parameter
```
- x -> input
- w -> weight // focussing on your data
- b -> bias // bias with your data
- O -> output // your output form formula
- act -> activation function
 - step function // y = 1 when x >= 0 || 0 when x < 0
 - sigh function // y = +1 when x >= 0 || -1 when x < 0
 - sigmoid function // y = 1/(1+(e^-x))
 - linear function // y = x
```

# perceptron learning
## w = w + α(t - o)x