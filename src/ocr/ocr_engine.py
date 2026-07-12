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

        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        if self.engine_type == "easyocr":
            # O EasyOCR utiliza redes neurais profundas (CRAFT + CRNN) e funciona melhor
            # com imagens em escala de cinza/coloridas sem binarização agressiva (limiarização/morfologia),
            # pois estas removem texturas e gradientes essenciais para os recursos convolucionais.
            # Redimensionamos em 3x com interpolação cúbica para melhorar a resolução espacial.
            return cv2.resize(
                plate_crop,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
            )

        # =====================================================
        # Pré-processamento clássico (Recomendado para Tesseract)
        # =====================================================
        plate = cv2.resize(
            plate_crop,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )
        gray = clahe.apply(gray)

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15
        )

        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )

        return binary

    # ------------------------------------------------------------------
    # Leitura da placa
    # ------------------------------------------------------------------
    def read_plate(self, plate_crop: np.ndarray) -> str:

        if plate_crop is None or plate_crop.size == 0:
            return ""

        processed = self.preprocess_plate(plate_crop)

        raw = ""

        if self.engine_type == "easyocr":

            results = self.reader.readtext(
                processed,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                paragraph=False,
                detail=1,
                width_ths=0.5,
                height_ths=0.5,
                decoder="beamsearch"
            )

            if results:
                raw = max(results, key=lambda x: x[2])[1]

        else:

            config = (
                "--psm 7 "
                "--oem 3 "
                "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            raw = pytesseract.image_to_string(
                processed,
                config=config
            )

        cleaned = self.clean_text(raw)

        if len(cleaned) == 7:
            cleaned = _apply_plate_mask(cleaned)

        return cleaned
