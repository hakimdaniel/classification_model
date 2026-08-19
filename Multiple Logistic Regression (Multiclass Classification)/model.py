import math

x1 = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

x2 = [18, 19, 20, 21, 22, 23, 24, 25, 27, 29, 31, 33, 35, 38, 40]

x3 = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 25, 30, 35]

y = [0, 0, 0, 0, 0,
     1, 1, 1, 1, 1,
     2, 2, 2, 2, 2]


# Class 0
w01 = 0
w02 = 0
w03 = 0
b0 = 0

# Class 1
w11 = 0
w12 = 0
w13 = 0
b1 = 0

# Class 2
w21 = 0
w22 = 0
w23 = 0
b2 = 0

loss1 = 0
loss2 = 0
loss3 = 0

learning_rate = 0.000001


for e in range(5000):

    print(f"(epoch {e + 1})===============")

    for i in range(len(y)):

        # Linear calculation

        z0 = (w01 * x1[i]) + (w02 * x2[i]) + (w03 * x3[i]) + b0

        z1 = (w11 * x1[i]) + (w12 * x2[i]) + (w13 * x3[i]) + b1

        z2 = (w21 * x1[i]) + (w22 * x2[i]) + (w23 * x3[i]) + b2


        # Softmax

        e0 = math.exp(z0)
        e1 = math.exp(z1)
        e2 = math.exp(z2)

        p0 = e0 / (e0 + e1 + e2)
        p1 = e1 / (e0 + e1 + e2)
        p2 = e2 / (e0 + e1 + e2)

        # Error
        y0 = 1 if y[i] == 0 else 0
        y1 = 1 if y[i] == 1 else 0
        y2 = 1 if y[i] == 2 else 0

        error0 = p0 - y0
        error1 = p1 - y1
        error2 = p2 - y2

        error0 = p0 - y0
        error1 = p1 - y1
        error2 = p2 - y2


        # Update Class 0

        w01 -= learning_rate * error0 * x1[i]
        w02 -= learning_rate * error0 * x2[i]
        w03 -= learning_rate * error0 * x3[i]
        b0  -= learning_rate * error0


        # Update Class 1

        w11 -= learning_rate * error1 * x1[i]
        w12 -= learning_rate * error1 * x2[i]
        w13 -= learning_rate * error1 * x3[i]
        b1  -= learning_rate * error1


        # Update Class 2

        w21 -= learning_rate * error2 * x1[i]
        w22 -= learning_rate * error2 * x2[i]
        w23 -= learning_rate * error2 * x3[i]
        b2  -= learning_rate * error2

        loss1 = -math.log(p0)
        loss2 = -math.log(p1)
        loss3 = -math.log(p2)

print("\n---------RESULT----------")
print("Trained model")
print("Class 1")
print("Weight 1:",w01)
print("Weight 2:",w02)
print("Weight 3:",w03)
print("Loss :",loss1)
print("\nClass 2")
print("Weight 1:",w11)
print("Weight 2:",w12)
print("Weight 3:",w13)
print("Loss :",loss2)
print("\nClass 3")
print("Weight 1:",w21)
print("Weight 2:",w22)
print("Weight 3:",w23)
print("Loss :",loss3)

def predict(x1,x2,x3):
    # Linear calculation
    
    z0 = (w01 * x1) + (w02 * x2) + (w03 * x3) + b0

    z1 = (w11 * x1) + (w12 * x2) + (w13 * x3) + b1

    z2 = (w21 * x1) + (w22 * x2) + (w23 * x3) + b2


    # Softmax

    e0 = math.exp(z0)
    e1 = math.exp(z1)
    e2 = math.exp(z2)

    p0 = e0 / (e0 + e1 + e2)
    p1 = e1 / (e0 + e1 + e2)
    p2 = e2 / (e0 + e1 + e2)

    probabilities = [p0, p1, p2]

    prediction = probabilities.index(max(probabilities))

    print("Prediction:", prediction)

predict(90,40,35)
