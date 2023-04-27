#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J marcin_wasowicz_master_thesis

## Liczba alokowanych węzłów
#SBATCH -N 1

## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=8

## Specyfikacja partycji GPU:
#SBATCH --mem-per-cpu=10GB
#SBATCH -A plgap2022-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --time=11:00:00

## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR

module load python/3.9.6-gcccore-11.2.0
module load cuda/11.6.0
module load cudnn/8.4.1.50-cuda-11.6.0

##pip uninstall -y dgl
##pip install gensim==4.2.0
##pip install nltk==3.7
##pip install torch==1.11.0
##pip install scikit-learn==1.2.0
##pip install dgl -f https://data.dgl.ai/wheels/cu116/repo.html
##pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
##pip install optuna==3.1.1
##pip install transformers==4.27.4

##python e2e/sst_dataset/prepare_constituency_sst_embeddings.py config/distilbert.json
##python e2e/sst_dataset/prepare_constituency_sst_embeddings.py config/glove.json
##python e2e/sst_dataset/prepare_constituency_sst_embeddings.py config/fasttext.json

##python e2e/sick_dataset/prepare_constituency_sick_embeddings.py config/distilbert.json
##python e2e/sick_dataset/prepare_constituency_sick_embeddings.py config/glove.json
##python e2e/sick_dataset/prepare_constituency_sick_embeddings.py config/fasttext.json

##python e2e/sst_dataset/train_sst_classifier.py config/lstm_n_ary.json
##python e2e/sick_dataset/train_sick_constituency_regressor.py config/lstm_n_ary.json

##python e2e/sst_dataset/train_sst_classifier.py config/lstm_sum.json
##python e2e/sick_dataset/train_sick_constituency_regressor.py config/lstm_sum.json

##python e2e/sst_dataset/train_sst_classifier.py config/lstm_single.json
##python e2e/sick_dataset/train_sick_constituency_regressor.py config/lstm_single.json

##python e2e/sst_dataset/train_sst_classifier.py config/gru_n_ary.json
##python e2e/sick_dataset/train_sick_constituency_regressor.py config/gru_n_ary.json

##python e2e/sst_dataset/train_sst_classifier.py config/gru_sum.json
##python e2e/sick_dataset/train_sick_constituency_regressor.py config/gru_sum.json

##python e2e/sst_dataset/train_sst_classifier.py config/gru_single.json
##python e2e/sick_dataset/train_sick_constituency_regressor.py config/gru_single.json

##python e2e/sst_dataset/train_sst_classifier.py config/mgu_n_ary.json
##python e2e/sick_dataset/train_sick_constituency_regressor.py config/mgu_n_ary.json

##python e2e/sst_dataset/train_sst_classifier.py config/mgu_sum.json
##python e2e/sick_dataset/train_sick_constituency_regressor.py config/mgu_sum.json

##python e2e/sst_dataset/train_sst_classifier.py config/mgu_single.json
##python e2e/sick_dataset/train_sick_constituency_regressor.py config/mgu_single.json
