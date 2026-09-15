"""한국어 자모 단위 토크나이저.

음절(가~힣, 11,172자)을 그대로 출력 단위로 쓰면 CTC 출력층이 너무 커지고
데이터가 적은 음절은 거의 학습되지 않는다. 그래서 음절을 초성/중성/종성으로
분해해 68개 정도의 토큰으로 학습하고, 디코딩 후 다시 음절로 조합한다.

초성 'ㄱ'과 종성 'ㄱ'은 서로 다른 토큰으로 둔다. 그래야 조합이 모호하지 않다.
"""

import re
import unicodedata

CHO = [chr(0x1100 + i) for i in range(19)]   # 초성 (Hangul Jamo 영역)
JUNG = [chr(0x1161 + i) for i in range(21)]  # 중성
JONG = [chr(0x11A8 + i) for i in range(27)]  # 종성 (받침 없음은 토큰 없음)

BLANK = 0


class KoreanJamoTokenizer:
    def __init__(self):
        # 0 = CTC blank, 1 = space, 이후 자모
        self.vocab = ["<blank>", " "] + CHO + JUNG + JONG
        self.index = {t: i for i, t in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    @staticmethod
    def normalize(text: str) -> str:
        """한글 음절과 공백만 남긴다. 숫자/영문은 녹음 문장에서 한글로 풀어 써야 한다."""
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[^가-힣 ]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def encode(self, text: str) -> list[int]:
        ids = []
        for ch in self.normalize(text):
            if ch == " ":
                ids.append(self.index[" "])
                continue
            code = ord(ch) - 0xAC00
            cho, jung, jong = code // 588, (code % 588) // 28, code % 28
            ids.append(self.index[CHO[cho]])
            ids.append(self.index[JUNG[jung]])
            if jong:
                ids.append(self.index[JONG[jong - 1]])
        return ids

    def decode(self, ids: list[int]) -> str:
        """자모 토큰 열을 음절로 조합한다. 짝이 맞지 않는 자모는 버린다."""
        tokens = [self.vocab[i] for i in ids if i != BLANK]
        out, i = [], 0
        while i < len(tokens):
            t = tokens[i]
            if t == " ":
                out.append(" ")
                i += 1
            elif t in CHO and i + 1 < len(tokens) and tokens[i + 1] in JUNG:
                cho, jung = CHO.index(t), JUNG.index(tokens[i + 1])
                jong = 0
                if i + 2 < len(tokens) and tokens[i + 2] in JONG:
                    jong = JONG.index(tokens[i + 2]) + 1
                out.append(chr(0xAC00 + cho * 588 + jung * 28 + jong))
                i += 3 if jong else 2
            else:
                i += 1
        return re.sub(r"\s+", " ", "".join(out)).strip()
