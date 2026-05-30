import cv2
import easyocr
import pytesseract
import re
import numpy as np

# ---------------------------------------------------------------------------
# Mapeamentos de correção por posição para placas brasileiras
#
# Formato antigo:  L L L N N N N   (ex: MLS5511)
# Formato Mercosul: L L L N L N N  (ex: ABC1D23)
#
# O OCR frequentemente troca:
#   letras por números: O↔0, I↔1, S↔5, B↔8, Z↔2, G↔6, Q↔0
#   números por letras: 0↔O, 1↔I, 5↔S, 8↔B, 2↔Z, 6↔G
# ---------------------------------------------------------------------------

# Caracteres que são letras mas parecem números
_NUM_TO_LETTER = str.maketrans('0158268479', 'OISBZGAHQ?')  # só se position for letra
# Caracteres que são números mas parecem letras
_LETTER_TO_NUM = str.maketrans('OISBZGAHQ', '015826649')   # só se position for dígito


def _fix_char(ch: str, expect_letter: bool) -> str:
    """Corrige um caractere OCR com base no tipo esperado na posição."""
    ch = ch.upper()
    if expect_letter:
        return ch.translate(_NUM_TO_LETTER)
    else:
        return ch.translate(_LETTER_TO_NUM)


def _is_mercosul(raw: str) -> bool:
    """Heurística para detectar se a placa está no formato Mercosul (AAA0A00)."""
    if len(raw) != 7:
        return False
    # Mercosul: posição 4 (índice 3) é número, posição 5 (índice 4) é letra
    # Antigo:   posições 4-7 (índices 3-6) são todos números
    return raw[4].isalpha() if raw[4].isascii() else False


def _apply_plate_mask(raw: str) -> str:
    """
    Aplica máscara de posição para corrigir confusões letra/número do OCR.

    Formato antigo:   L L L N N N N  (posições 0,1,2 = letra; 3,4,5,6 = dígito)
    Formato Mercosul: L L L N L N N  (posições 0,1,2 = letra; 3 = dígito;
                                       4 = letra; 5,6 = dígito)
    """
    if len(raw) < 7:
        return raw  # Muito curto — não tenta corrigir

    mercosul = _is_mercosul(raw)

    if mercosul:
        mask = [True, True, True, False, True, False, False]  # True = espera letra
    else:
        mask = [True, True, True, False, False, False, False]

    corrected = []
    for i, ch in enumerate(raw[:7]):
        if i < len(mask):
            corrected.append(_fix_char(ch, expect_letter=mask[i]))
        else:
            corrected.append(ch)

    return ''.join(corrected)


class OCREngine:
    def __init__(self, engine_type='easyocr'):
        self.engine_type = engine_type
        if engine_type == 'easyocr':
            # Português + inglês; GPU se disponível
            self.reader = easyocr.Reader(['pt', 'en'], gpu=True)

    # ------------------------------------------------------------------
    # Limpeza de texto bruto
    # ------------------------------------------------------------------
    def clean_text(self, text: str) -> str:
        """Remove tudo que não é letra ou dígito e converte para maiúsculo."""
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    # ------------------------------------------------------------------
    # Pré-processamento da imagem da placa
    # ------------------------------------------------------------------
    def preprocess_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Pipeline de pré-processamento otimizado para placas brasileiras.

        1. Upscaling 3× (melhora resolução para o OCR)
        2. Conversão para cinza
        3. CLAHE (equalização adaptativa de histograma — melhora contraste)
        4. Denoising leve (preserva bordas de caracteres)
        5. Threshold de Otsu (binarização global — mais estável que o adaptativo
           para placas com fundo uniforme)
        """
        h, w = plate_crop.shape[:2]
        # 1. Upscaling
        plate = cv2.resize(plate_crop, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

        # 2. Cinza
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        # 3. CLAHE — melhora contraste local sem destruir bordas finas
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

        # 4. Denoising leve
        gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

        # 5. Threshold de Otsu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    # ------------------------------------------------------------------
    # Leitura da placa
    # ------------------------------------------------------------------
    def read_plate(self, plate_crop: np.ndarray) -> str:
        """
        Lê o texto de uma imagem de placa recortada.

        Etapas:
          1. Pré-processamento da imagem
          2. OCR (EasyOCR ou Tesseract)
          3. Limpeza do texto bruto
          4. Correção posicional letra/número (máscara de placa brasileira)
        """
        if plate_crop is None or plate_crop.size == 0:
            return ""

        processed = self.preprocess_plate(plate_crop)

        raw = ""
        if self.engine_type == 'easyocr':
            results = self.reader.readtext(processed)
            if results:
                # Pega a leitura com maior confiança
                raw = max(results, key=lambda x: x[2])[1]
        else:
            config = (
                '--psm 7 '
                '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            raw = pytesseract.image_to_string(processed, config=config)

        cleaned = self.clean_text(raw)

        # Aplica correção posicional se o texto tiver comprimento de placa válido (7)
        if len(cleaned) == 7:
            cleaned = _apply_plate_mask(cleaned)

        return cleaned
