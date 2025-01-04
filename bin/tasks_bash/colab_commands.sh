!git clone https://limingzhou2004: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@github.com/limingzhou2004/pytools.git

!cd pytools && git checkout prod && pip install --ignore-installed -r requirements.txt && python -m pytools.weather_task -cfg /content/drive/MyDrive/sites/config/albany_prod_colab.toml task_3 --flag cv -ind 2 -sb fit -mn prod2_v59 -yr 2018-2024 -nworker 7
