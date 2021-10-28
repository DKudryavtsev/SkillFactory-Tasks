"""The game of guessing a number
Computer generates and guesses 1000 numbers
"""
import numpy as np


def center_predict(number:int=1) -> int:
    """Computer guesses the generated number as the center of the range

    Args:
        number (int, optional): the generated number. Defaults to 1.

    Returns:
        int: the number of attempts.
    """
    
    count = 0          # number of attempts
    lower_bound = 1    # initial lower bound
    upper_bound = 100  # initial upper bound
    
    while True:
        count += 1
        predict_number = (lower_bound+upper_bound) // 2  # a possible number
        
        if predict_number > number:
            upper_bound = predict_number - 1
        elif predict_number < number:
            lower_bound = predict_number + 1
        else:
            break  # cycle break if the match
    
    return count


def score_game(center_predict) -> int:
    """Calculating the mean value of guessing attempts for 1000 games

    Args:
        center_predict (func): the guessing function

    Returns:
        int: the mean value of attempts
    """
    
    random_array = np.random.randint(1, 101, size=1000)  # list of generated numbers
    count_ls = []                                        # number of attempts for each generated number
            
    for number in random_array:             
        count_ls.append(center_predict(number))
    score = int(np.mean(count_ls))
    
    print(f'Your algorithm guesses a number in {score} attempts, on the average')
    return score


if __name__ == '__main__':
    score_game(center_predict)