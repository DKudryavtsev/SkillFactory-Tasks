"""The game of guessing a number"""
import numpy as np

number = np.random.randint(1, 101)  # generated number
count = 0                           # number of attempts

while True:
    count += 1
    predict_number = int(input('Guess the number from 1 to 100: '))
    
    if predict_number > number:
        print('The number must be less')
    elif predict_number < number:
        print('The number must be greater')
    else:
        print(f'You\'ve guessed the number {number} in {count} attempts')
        break  # end of the game