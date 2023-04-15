#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J marcin_wasowicz_master_thesis

## Liczba alokowanych węzłów
#SBATCH -N 1

## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=24

## Specyfikacja partycji CPU:
#SBATCH --mem-per-cpu=5GB
#SBATCH -A plgap2022-cpu
#SBATCH -p plgrid
#SBATCH --time=10:00:00

## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR

module load python/3.9.6-gcccore-11.2.0

pip uninstall dgl
pip install dgl==0.9.1
pip install gensim==4.2.0
pip install nltk==3.7
pip install torch==1.11.0
pip install scikit-learn==1.2.0
pip install optuna==3.1.1

##python e2e/sst_dataset/prepare_constituency_sst_embeddings.py config/sst_classification.json
##python e2e/sick_dataset/prepare_constituency_sick_embeddings.py config/sick_regression.json
##python e2e/sst_dataset/train_sst_classifier.py config/sst_classification.json
##python e2e/sick_dataset/train_sick_constituency_regressor.py config/sick_regression.json