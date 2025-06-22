export TF_ENABLE_ONEDNN_OPTS=0

# Recebe o nome do arquivo Python como argumento e executa mantendo a variável de ambiente
if [ $# -eq 0 ]; then
    echo "Uso: $0 <arquivo_python>"
    exit 1
fi

py "$1"