from src.utils.ufpr_parser import parse_annotation, load_dataset_split
from pathlib import Path

# Testa o parser em um arquivo
txt = Path("UFPR-ALPR dataset/testing/track0092/track0092[02].txt")
result = parse_annotation(txt)
print("=== Teste do Parser ===")
print("Placa (ground truth):", result["plate_text"])
print("BBox Veiculo:", result["vehicle_bbox"])
print("BBox Placa  :", result["plate_bbox"])
print("Cantos Placa:", result["plate_corners"])

# Testa o carregamento de amostras
print()
samples = load_dataset_split(Path("UFPR-ALPR dataset"), split="testing", limit=5)
print("=== Primeiras 5 amostras do testing ===")
for s in samples:
    print(" ", s["image_path"].name, "-> placa:", s["annotation"]["plate_text"])
