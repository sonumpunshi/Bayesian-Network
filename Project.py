import sys
from collections import defaultdict


def read_training_data(filename):
    dataset = []
    with open(filename, 'r') as file:
        for line in file:
            record = [int(value) for value in line.strip().split()]
            dataset.append(record)
    return dataset


# Calculate probabilities from the data
def calculate_probabilities(dataset):
    counts = {
        "B": [0, 0],
        "G_given_B": {0: [0, 0], 1: [0, 0]},
        "C": [0, 0],
        "F_given_G_C": {(0, 0): [0, 0], (0, 1): [0, 0], (1, 0): [0, 0], (1, 1): [0, 0]}
    }

    # Count occurrences of variables
    for record in dataset:
        b, g, c, f = record
        counts["B"][b] += 1
        counts["G_given_B"][b][g] += 1
        counts["C"][c] += 1
        counts["F_given_G_C"][(g, c)][f] += 1

    num_data_points = len(dataset)
    # Calculate probabilities
    probabilities = {
        "P(B)": [count / num_data_points for count in counts["B"]],
        "P(G|B)": {key: [count / sum(val) for count in val] for key, val in counts["G_given_B"].items()},
        "P(C)": [count / num_data_points for count in counts["C"]],
        "P(F|G,C)": {key: [count / sum(val) for count in val] for key, val in counts["F_given_G_C"].items()},
    }
    return probabilities


# Display calculated probabilities
def display_probabilities(probabilities):
    print("P(B):")
    print(f"  False: {probabilities['P(B)'][0]}, True: {probabilities['P(B)'][1]}\n")

    print("P(G|B):")
    print(f"  B=False: [False: {probabilities['P(G|B)'][0][0]}, True: {probabilities['P(G|B)'][0][1]}]")
    print(f"  B=True:  [False: {probabilities['P(G|B)'][1][0]}, True: {probabilities['P(G|B)'][1][1]}]\n")

    print("P(C):")
    print(f"  False: {probabilities['P(C)'][0]}, True: {probabilities['P(C)'][1]}\n")

    print("P(F|G,C):")
    for g in range(2):
        for c in range(2):
            print(
                f"  G={bool(g)}, C={bool(c)}: [False: {probabilities['P(F|G,C)'][(g, c)][0]}, True: {probabilities['P(F|G,C)'][(g, c)][1]}]")


# Joint probability distribution
def jpd(probabilities, b_val, g_val, c_val, f_val):
    b = 1 if b_val == "Bt" else 0
    g = 1 if g_val == "Gt" else 0
    c = 1 if c_val == "Ct" else 0
    f = 1 if f_val == "Ft" else 0

    p_b = probabilities["P(B)"][b]
    p_g_given_b = probabilities["P(G|B)"][b][g]
    p_c = probabilities["P(C)"][c]
    p_f_given_g_c = probabilities["P(F|G,C)"][(g, c)][f]

    prob = p_b * p_g_given_b * p_c * p_f_given_g_c
    return prob


# Conditional probability calculation
def conditional_probability(probabilities, query, evidence):
    query_vars = ["B", "G", "C", "F"]
    query_values = [1 if v == "t" else 0 for v in query]
    evidence_values = [1 if v == "t" else 0 for v in evidence]
    numerator = 0
    denominator = 0
    for b in range(2):
        for g in range(2):
            for c in range(2):
                for f in range(2):
                    prob = probabilities["P(B)"][b] * probabilities["P(G|B)"][b][g] * probabilities["P(C)"][c] * \
                           probabilities["P(F|G,C)"][(g, c)][f]

                    current_values = {"B": b, "G": g, "C": c, "F": f}

                    if all(query_values[i] == current_values[query_vars[i]] for i in range(len(query))):
                        if all(evidence_values[i] == current_values[query_vars[i+2]] for i in range(len(evidence))):
                            numerator += prob

                    if all(evidence_values[i] == current_values[query_vars[i+2]] for i in range(len(evidence))):
                        denominator += prob
    return numerator / denominator


def inf_probability(probabilities, query):
    query_vars = ["B", "G", "C", "F"]
    query_values = [None if v is None else (1 if v[1] == "t" else 0) for v in query]
    total_prob = 0
    for b in range(2):
        for g in range(2):
            for c in range(2):
                for f in range(2):
                    current_values = [b, g, c, f]
                    if all(query_values[i] is None or query_values[i] == current_values[i] for i in range(4)):
                        prob = probabilities["P(B)"][b] * probabilities["P(G|B)"][b][g] * probabilities["P(C)"][c] * \
                               probabilities["P(F|G,C)"][(g, c)][f]
                        total_prob += prob
    return total_prob


# Main function
if __name__ == "__main__":
    training_data_file = sys.argv[1]
    dataset = read_training_data(training_data_file)
    probabilities = calculate_probabilities(dataset)

    if len(sys.argv) == 2:
        # Task 1
        display_probabilities(probabilities)
    else:
        variables = ["B", "G", "C", "F"]
        values = [None] * 4
        for arg in sys.argv[2:]:
            index = variables.index(arg[0])
            values[index] = arg

        query = [value for value in values if value is not None]

        if len(query) == 2:
            # Task 2: Calculate P(B=t, F=f) using inference by enumeration
            marginal_prob = conditional_probability(probabilities, query, [])
            print(f"Probability: {marginal_prob}")
        elif len(query) == 4:
            # Task 2: Calculate P(B=t, G=f, C=t, F=f) using joint probability distribution
            b_val, g_val, c_val, f_val = query
            joint_prob = jpd(probabilities, b_val, g_val, c_val, f_val)
            print(f"P({b_val}, {g_val}, {c_val}, {f_val}) = {joint_prob}")
