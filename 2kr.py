import itertools
import pandas as pd
import numpy as np

import itertools
import pandas as pd

label = ['A', 'B', 'C', 'D', 'F']

def generate_sign_table(k):
    levels = [-1, 1]
    combinations = list(itertools.product(levels, repeat=k))
    
    # Inverte os dados ANTES de nomear
    combinations = [list(c)[::-1] for c in combinations]
    
    factor_labels = label[:k]
    df = pd.DataFrame(combinations, columns=factor_labels)

    # Agora não precisa mais fazer df.iloc[:, ::-1]
    
    # Adiciona colunas de interação
    for i in range(2, k+1):
        for combo in itertools.combinations(factor_labels, i):
            col_name = ''.join(combo)
            df[col_name] = df[list(combo)].prod(axis=1)

    return df



def get_responses(df, r):
    y_columns = []
    for i in range(r):
        y_column = []
        print(f"\n--- Coletando respostas da repetição {i+1} ---")
        for idx, row in df.iterrows():
            levels = ', '.join([f"{col}={row[col]}" for col in df.columns[:k]])
            y = float(input(f"Digite a resposta para {levels}: "))
            y_column.append(y)
        y_columns.append(y_column)
    return np.array(y_columns).T  # shape: (num_combinations, r)

def calculate_effects(df, Y_avg):
    X = df.to_numpy()
    effects = (X.T @ Y_avg) / len(Y_avg)
    return effects

def compute_variation(Y, effects, r, k):
    # Soma dos quadrados dos efeitos (SSe)
    num_runs = 2 ** k
    SS_effects = num_runs * r * sum(e**2 for e in effects)

    # Média das respostas
    Y_avg = Y.mean(axis=1)

    # Erro experimental (SSErro)
    SS_error = ((Y - Y_avg[:, None]) ** 2).sum()

    # Soma total dos quadrados (SST) via fórmula do enunciado
    SST = SS_effects + SS_error

    # Porção de variação explicada por cada efeito
    variation_explained = [
        (num_runs * r * e**2) / SST for e in effects
    ]

    return SST, SS_effects, SS_error, variation_explained

# Interface principal
k = int(input("Digite o número de fatores (2 a 5): "))
r = int(input("Digite o número de replicações (1 a 3): "))

# Validação básica
assert 2 <= k <= 5, "Número de fatores deve ser entre 2 e 5."
assert 1 <= r <= 3, "Número de replicações deve ser entre 1 e 3."

# Geração da tabela e coleta de dados
df_signals = generate_sign_table(k)
Y = get_responses(df_signals, r)
Y_avg = Y.mean(axis=1)
effects = calculate_effects(df_signals, Y_avg)
SST, SS_eff, SS_err, variation_explained = compute_variation(Y, effects, r, k)

# Monta a tabela final com efeitos e % de explicação
final_df = df_signals.copy()
final_df["Y médio"] = Y_avg
effects_labels = list(df_signals.columns)
result_df = pd.DataFrame({
    "Efeito": effects_labels,
    "Valor": effects,
    "% Var Explicada": [v * 100 for v in variation_explained]
})

final_df = pd.concat([final_df, pd.DataFrame(Y, columns=[f"Y{i+1}" for i in range(r)])], axis=1)

# Exibe resultados
print("\n========== Tabela Final ==========")
print(final_df)

print("\n========== Efeitos e Variação Explicada ==========")
print(result_df)

print("\n========== Resumo ==========")
print({
    "Soma Total dos Quadrados (SST)": SST,
    "Soma dos Quadrados dos Efeitos (SS efeitos)": SS_eff,
    "Erro experimental (SS erro)": SS_err
})
