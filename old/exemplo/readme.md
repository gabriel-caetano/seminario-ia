# Exemplo de Uso de Modelos em Python

Este diretório contém exemplos de uso de diferentes scripts Python para tarefas de aprendizado de máquina com dados tabulares.

## Passo a Passo para Execução

### 1. Instale o Python (recomendado Python 3.8+)
Certifique-se de ter o Python instalado em seu sistema. Baixe em: https://www.python.org/downloads/

### 2. (Opcional) Crie um ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

### 3. Instale as dependências necessárias
Cada script pode exigir dependências diferentes. Para instalar as dependências principais, execute:
```bash
pip install pandas scikit-learn tabpfn tensorflow
```
Se algum script exigir dependências adicionais, consulte o início do arquivo ou um requirements.txt correspondente.

### 4. Prepare o dataset
Certifique-se de que os arquivos `dataset.csv`, `dataset_source.csv` e `dataset_target.csv` estejam presentes neste diretório.

### 5. Execute o script desejado
Para rodar qualquer script Python deste diretório, utilize o comando:
```bash
python nome_do_arquivo.py
```
Exemplo:
```bash
python transfer.py
```

### 6. Sobre cada script
- `split_ds.py`: Apenas separa os dados do domínio source e target, gerando os arquivos de dataset correspondentes.
- `source.py` e `target.py`: Testam o treinamento independente dos domínios source e target, respectivamente.
- `transfer.py`: Realiza um teste de transfer learning, treinando o modelo no domínio source e avaliando no domínio target.

### 7. Interpretação dos Resultados do transfer.py
O arquivo `transfer.py` imprime métricas como loss e acurácia durante o treinamento e validação, mostrando a performance do modelo ao transferir o aprendizado do domínio source para o domínio target. Os resultados geralmente incluem:
- **Loss**: Erro do modelo durante o treinamento e validação.
- **Accuracy**: Proporção de acertos do modelo.
Essas métricas ajudam a avaliar se o conhecimento aprendido no domínio source é útil para o domínio target.
