from pyemd import emd
import numpy as np

"""
All of our model are evaluated by fasttext
"""

def calculate_emd(input_distribution, output_distribution):   
    N = len(input_distribution)
    distance_matrix = np.ones((N, N))
    return emd(input_distribution, output_distribution, distance_matrix)

def account_for_direction(input_target_style_probability, output_target_style_probability):
    if output_target_style_probability >= input_target_style_probability:
        return 1
    return -1

def calculate_direction_corrected_emd(input_distribution, output_distribution, target_style_class): 
    emd_score = calculate_emd(input_distribution, output_distribution)
    direction_factor = account_for_direction(input_distribution[target_style_class], output_distribution[target_style_class])
    return emd_score * direction_factor

def calculate_STIs(sequences_input, sequences_output, target_styles, model):
    def get_class_probs(sequence, model):
        labels, ps = model.predict(sequence, k=len(model.labels))
        pairs = list(zip(labels, ps.tolist()))
        pairs.sort(key=lambda e: e[0])
        return np.array([p for _, p in pairs])
    input_probs = [get_class_probs(s, model) for s in sequences_input]
    output_probs = [get_class_probs(s, model) for s in sequences_output]
    return [calculate_direction_corrected_emd(p_i, p_o, tgt) \
        for p_i, p_o, tgt in zip(input_probs, output_probs, target_styles)]


if __name__ == "__main__":
    import fasttext
    inputs = ["and the cleaning is way over priced .", "i hate the cornbread appetizer ."]
    outputs = ["and the cleaning is way perfectly priced .", "i love the cornbread appetizer ."]
    model = fasttext.load_model("../eval_dump/model_yelp.bin")
    print(calculate_STIs(inputs, outputs, [1, 0], model))
