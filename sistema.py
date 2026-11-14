import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("--- MOTOR DE GÉMEOS MINERALÓGICOS (RIGOR MÁXIMO) ---")

# 1. Dados de referência oficiais (Lua e Marte)
perfis_referencia = {
    'Planet_Ref': ['alto_ti', 'alto_ti', 'alto_ti', 'alto_ti', 'baixo_ti', 'baixo_ti', 'baixo_ti', 'baixo_ti', 'baixo_ti', 'marte'],
    'SIO2': [41.0, 39.8, 37.8, 38.6, 46.3, 43.6, 45.3, 48.0, 45.2, 45.6],
    'TIO2': [11.3, 10.5, 13.0, 8.8, 4.0, 2.6, 4.7, 1.8, 2.4, 0.55],
    'AL2O3': [9.5, 10.4, 8.8, 6.3, 10.7, 7.9, 10.0, 9.4, 8.6, 10.6],
    'MNO': [0.25, 0.30, 0.27, 0.27, 0.27, 0.28, 0.28, 0.30, 0.30, 0.38],
    'MGO': [7.03, 6.7, 8.4, 14.4, 6.5, 14.9, 7.0, 8.7, 10.3, 10.7],
    'CAO': [11.0, 11.1, 10.7, 7.7, 11.8, 8.3, 11.4, 10.4, 9.8, 7.50],
    'NA20': [0.51, 0.40, 0.36, 0.36, 0.28, 0.23, 0.29, 0.32, 0.31, 3.0],
    'K20': [0.36, 0.06, 0.05, 0.09, 0.07, 0.05, 0.06, 0.06, 0.04, 0.15],
    
}

df_ref = pd.DataFrame(perfis_referencia) 
# 2. As vossas amostras do Paraná para teste
amostras_parana = pd.DataFrame({
    'Amostra_ID': ['Sample_PR_01', 'Sample_PR_02'],
    'SIO2': [41.9, 35.1], 
    'TIO2': [11.6, 44.9],
    'AL2O3': [9.1,  14.8],
    'MNO': [0.1, 0.3],
    'MGO': [7.0, 0.4],
    'CAO': [11, 0.3],
    'NA20': [0.5, 0.2],
    'K20': [0.3, 0.2]
})

features = ['SIO2', 'TIO2', 'AL2O3','MNO', 'MGO', 'CAO', 'NA20', 'K20']

# 3. Normalizar os dados para que todas as percentagens pesem igual na matemática
scaler = StandardScaler()
scaler.fit(pd.concat([df_ref[features], amostras_parana[features]]))

ref_scaled = scaler.transform(df_ref[features])
parana_scaled = scaler.transform(amostras_parana[features])

# --- DEFINIÇÃO DO RIGOR (Mínimo de 90% de similaridade exigido) ---
SIMILARIDADE_MINIMA = 90.0

print("\n" + "="*70)
print(f" AVALIAÇÃO DE RIGOR MÁXIMO (EXIGÊNCIA: {SIMILARIDADE_MINIMA}%) ")
print("="*70)

for i, row in amostras_parana.iterrows():
    amostra_id = row['Amostra_ID']
    vetor_amostra = parana_scaled[i].reshape(1, -1)
    
    # Calcular a distância para a Lua e para Marte
    distancias = np.linalg.norm(ref_scaled - vetor_amostra, axis=1)
    
    # Converter a menor distância num Score de Similaridade percentual (0 a 100%)
    menor_distancia = np.min(distancias)
    score_similaridade = max(0, 100 - (menor_distancia * 25)) # Fator de escala rigoroso
    
    indice_alvo = np.argmin(distancias)
    planeta_alvo = df_ref.loc[indice_alvo, 'Planet_Ref']

    # Aplicar o filtro de rigor estrito
    if score_similaridade < SIMILARIDADE_MINIMA:
        print(f"🔹 {amostra_id} ➔ ⚠️ REJEITADO (Similaridade com {planeta_alvo}: {score_similaridade:.1f}% < Mínimo de {SIMILARIDADE_MINIMA}%)")

    else:
        status = f"✅ GÉMEO {planeta_alvo.upper()} VALIDADO"
        print(f"🔹 {amostra_id} ➔ {status} (Similaridade Exata: {score_similaridade:.1f}%)")

print("="*70)
