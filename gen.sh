mkdir -p notebooks
wget -O notebooks/main.ipynb 'https://docs.google.com/uc?export=download&id=1XJxV70SruNKlxs2sOLmAOEiGlic-a7k2'
jupyter nbconvert --to script notebooks/main.ipynb
python process_notebook_code.py < notebooks/main.txt > colabcode.py
rm notebooks/main.txt