"""
CSE 150A/250A SP 25 HW 1

This file is meant to give you a template for implmenting hangman in question 1.6. You are not required to use the same implementation in your solution.

To run, call "python hangman.py".
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from tabulate import tabulate

def compute_prior(word_counts):
    """
    TODO
    Computes the prior probability for every word in the corpus. In other words, computing P(W=w).
    
    Args:
        word_counts (pd.DataFrame): DataFrame containing words and their counts

    Returns:
        pd.Series: Prior probabilities for all words in the corpus
    """
    total = word_counts['Count'].sum()
    prior = word_counts['Count'] / total
    return prior 

def get_prior(word,word_counts):
    """
    TODO
    Gets the prior probability for a given word from the dataframe.
    
    Args:
        word (str): Word to get prior probability for.
        word_counts (pd.DataFrame): DataFrame containing words, counts, and prior probabilities.

    Returns:
        float: Prior probability for a given word.
    """
    row = word_counts[word_counts['Word'] == word]
    if not row.empty:
        return row['Prior'].values[0]
    else: 
        return 0.0

def check_letter(l, w):
    """
    TODO
    Checks if the given letter is in the given word or P(Li = l for some i in {1,2,3,4,5} | W).
        
    Args:
        l (str): Letter of interest.
        w (str): Word to check letter in.
    Returns:
        bool: True if l is in w, False otherwise.
    """
    return l in w

def check_evidence(evidence,w):
    """
    TODO
    Checks if it's possible that the evidence supports the given word. In other words, its trying to compute P(E|W=w).

    Evidence is a tuple containing two strings, first is the guessed word so far with correct letters 
    and the second being all incorrect letters.
    
    Args:
        evidence (tuple): A tuple containing two strings
        w (str): Word to be checked

    Returns:
        bool: True if its possible, False otherwise.
    """
    correct, incorrect = evidence

    for i in range(len(w)):
        if correct[i] != '-' and w[i] != correct[i]:
            return False

    # Check that no incorrect letters appear in the word
    for letter in incorrect:
        if letter in w:
            return False

    return True

def compute_posterior_denominator(evidence,word_counts):
    """
    TODO
    Computes the probability of the evidence occurring or P(E). This is used in the calculation of the posterior probability.
    
    Args:
        evidence (tuple): A tuple containing two strings
        word_counts (pd.DataFrame): DataFrame containing words, counts, and prior probabilities.

    Returns:
        float: Probability of evidence.
    """
    denom = 0
    length = range(len(word_counts))
    for i in length: 
        w = word_counts.loc[i, 'Word']
        p = word_counts.loc[i, 'Prior']
        if check_evidence(evidence, w):
            denom += p
    return denom

def compute_posterior(evidence, word, word_counts, denominator):
    """
    TODO
    Computes the posterior probability or P(W=w|E).

    Should be computing the denominator separately for faster runtime.
    
    Args:
        evidence (tuple): A tuple containing two strings
        word (str): A given word to compute posterior for.
        word_counts (pd.DataFrame): DataFrame containing words, counts, and prior probabilities.
        Denominator (float): Denominator of the posterior probability computed earlier

    Returns:
        float: Probability of evidence.
    """
    numerator = check_evidence(evidence,word) * get_prior(word, word_counts)
    if denominator > 0:
        return numerator / denominator
    else: 
        return 0

def predictive_probability(evidence, word_counts,denominator):
    """
    TODO
    Computes the probability for each letter being in the word given the evidence or P(Li = l for some i in {1,2,3,4,5} | E)
    
    Args:
        evidence (tuple): A tuple containing two strings
        word_counts (pd.DataFrame): DataFrame containing words, counts, and prior probabilities.
        denominator (float): Denominator of posterior probability.

    Returns:
        lst: A list of probabilities for each letter.
    """
    val_letters, inval_letters = evidence 
    probs = {} 
    for i in range(len(word_counts)):
        w = word_counts.loc[i, 'Word']
        p = compute_posterior(evidence, w, word_counts, denominator) # posterior

        if p > 0: 
            for k in range(len(w)):
                letter = w[k]
                if letter in val_letters or letter in inval_letters:
                    continue
                if letter not in probs: 
                    probs[letter] = 0
                probs[letter] += p 
    return [probs.get(chr(i), 0.0) for i in range(65,91)] 
        

        
def predict_character(evidence, word_counts,denominator):
    """
    TODO
    Generates the prediction for the next best guess with the associated probability.
    
    Args:
        evidence (tuple): A tuple containing two strings
        word_counts (pd.DataFrame): DataFrame containing words, counts, and prior probabilities.
        denominator (float): Denominator of the posterior probability.

    Returns:
        tuple: Predicted character and associated probability.
    """
    letters_p = predictive_probability(evidence, word_counts, denominator) # letters probability 
    max_index = letters_p.index(max(letters_p))
    max_letter = chr(max_index + ord('A')) # max_index (0,1) + ascii char A or 65 
    best_guess = letters_p[max_index]
   
    return (max_letter, best_guess) # return best guess 



if __name__ == "__main__":    
    
    # TODO: Fill in correct file path
    file_path = "hw1_word_counts_05.txt"
    
    empty_word = "-----"
    Evidence = [(empty_word, ""),
               (empty_word, "EA"),
                ("A---S", ""),
                ("A---S","I"),
                ("--O--","AEMNT"),
                (empty_word,"EO"),
                ("D--I-",""),
                ("D--I-","A"),
                ("-U---","AEIOS")
               ]
    
    ### DO NOT MODIFY BELOW THIS LINE ###
    assert os.path.exists(file_path), f"File not found: {file_path}"
    

    word_counts = pd.read_csv(file_path,header=None,sep = ' ')
    word_counts = word_counts.rename(columns={0:'Word',1:'Count'})
    
    word_counts['Prior'] = compute_prior(word_counts)

    output = []
    pbar = tqdm(Evidence)
    for e in (Evidence):
        corr, incorr = e
        incorr = "{" + incorr + "}"
        pbar.set_description(f"Processing Evidence: '{e}'")
        char,prob = predict_character(e,word_counts,compute_posterior_denominator(e,word_counts))
        output += [(corr,incorr,char,prob)]
        pbar.update(1)

    output = pd.DataFrame(output,columns = ["Correctly Guessed", "Incorrectly Guessed","Character", "Probability"])
    print(print(tabulate(output, headers='keys', tablefmt='psql')))
    
        
