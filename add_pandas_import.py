import json
from pathlib import Path

notebook_path = Path("c:/Users/edvan/OneDrive/Documentos/TCC/Analise_e_Treinamento_TCC.ipynb")

if notebook_path.exists():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "import pandas as pd" not in source:
                cell['source'].insert(0, "import pandas as pd\n")
                break
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("Notebook atualizado com import pandas!")
else:
    print("Notebook não encontrado.")
