import math

# temperature (c°)
x1 = [
    36.2, 36.4, 36.5, 36.7, 36.8,
    37.0, 37.1, 37.2, 37.3, 37.5,

    38.0, 38.1, 38.2, 38.4, 38.5,
    38.7, 38.8, 39.0, 39.1, 39.2
]

# heart rate (bpm)
x2 = [
    65, 68, 72, 75, 78,
    80, 82, 85, 88, 90,

    92, 95, 98, 100, 102,
    105, 108, 110, 112, 115
]

# output
y = [
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
]

w1 = 0
w2 = 0
b = 0
learning_rate = 0.0001

print("Start training")
for e in range(5000):
    print(f"\n(epoch {e+1})===============")
    for i in range(len(y)):
        
        z = (w1 * x1[i]) + (w2 * x2[i]) + b
        prediction = 1/(1+math.exp(-z))
        
        loss = -(y[i] * math.log(prediction) + (1-y[i]) * math.log(1 - prediction))
        
        dw1 = (prediction - y[i]) * x1[i]
        dw2 = (prediction - y[i]) * x2[i]
        db =  (prediction - y[i])
        
        w1 = w1 - (learning_rate * dw1)
        w2 = w2 - (learning_rate * dw2)
        b = b - (learning_rate * db)
    
print("\n--------RESULT---------")
print("Learning rate:",learning_rate)
print("Weight 1:",w1)
print("Weight 2:",w2)
print("Bias:",b)
print("Loss:",loss)

def predict(x1,x2):
    z = (w1*x1) + (w2*x2) + b
    #sigmoid
    prediction = 1/(1+math.exp(-z))
    return prediction
    
def classify(score):
    if score >= 0.5:
        return "Demam"
    else:
        return "Tak Demam"

print("\nTest model")
print("Temperature (C°) | Heart rate (bpm)")


print("\n-------- TEST MODEL --------")

test = predict(36.2, 65)
print("\nInput 36.2 65:",test)
print("Classification:",classify(test))

test = predict(38.1, 95)
print("\nInput 38.1 95:",test)
print("Classification:",classify(test))

test = predict(34, 63)
print("\nInput 34 63:",test)
print("Classification:",classify(test))

test = predict(37.4, 93)
print("\nInput 37.4 93:",test)
print("Classification:",classify(test))
