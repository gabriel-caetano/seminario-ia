# Exemplo de Uso do TabPFN

Este diretório contém um exemplo de uso do modelo TabPFN para classificação binária utilizando um dataset tabular.

## Passo a Passo para Execução

### 1. Instale o Python (recomendado Python 3.8+)
Certifique-se de ter o Python instalado em seu sistema. Você pode baixar em: https://www.python.org/downloads/

### 2. Crie um ambiente virtual (opcional, mas recomendado)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

### 3. Instale as dependências
No diretório do exemplo, execute:
```bash
pip install tabpfn pandas scikit-learn
```

### 4. Coloque o arquivo `dataset.csv` no mesmo diretório
O arquivo `dataset.csv` deve conter os dados tabulares, com a coluna alvo chamada `death`.

### 5. Execute o exemplo
```bash
python tabpfn_exemplo.py
```

### 6. Interpretação dos Resultados
O script irá imprimir três métricas principais:
- **ROC AUC**: Mede a capacidade do modelo de distinguir entre as classes. Quanto mais próximo de 1, melhor.
- **Accuracy**: Proporção de acertos do modelo.
- **Log Loss**: Mede a performance do modelo considerando as probabilidades previstas. Quanto menor, melhor.

Exemplo de saída:
```
ROC AUC: 0.92
Accuracy 0.85
Log Loss: 0.32
```

## Observações
- Se você não possuir GPU, o TabPFN funcionará normalmente na CPU, porém pode ser mais lento.
- Para mais informações sobre o TabPFN, consulte: https://github.com/PriorLabs/TabPFN
