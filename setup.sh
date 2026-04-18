echo '
# Med FL project setup — run: medfl
alias medfl="cd /gpfs/home/s001/ssubram7/projects/med-learning-federated-system && \
  conda activate fed-learning-env && \
  export ISIC_DATA_ROOT=\$(python load_data_scratch.py --path-only) && \
  echo \"ISIC_DATA_ROOT=\$ISIC_DATA_ROOT\""
' >> ~/.bashrc
source ~/.bashrc
