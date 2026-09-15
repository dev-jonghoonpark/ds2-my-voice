# ds2-my-voice

**DeepSpeech2(DS2)를 PyTorch로 직접 구현하고, 공개 한국어 데이터로 사전학습한 다음 내 목소리로 파인튜닝해 보는 학습용 데모.**

- 🔗 **데모 페이지**: https://dev-jonghoonpark.github.io/ds2-my-voice/
- 📦 **GitHub 레포**: https://github.com/dev-jonghoonpark/ds2-my-voice
- 📚 **시리즈**: [how-ai-works](https://github.com/dev-jonghoonpark/how-ai-works) — 음성 인식 (Deep Speech 2) 섹션

목표는 최고 성능이 아니라 음성인식 파이프라인 전체(녹음 → 특징 추출 → CTC 학습 → 디코딩 → 평가)를 손으로 만져 보는 것입니다.
최종 결과물은 "사전학습 모델 vs 내 목소리로 파인튜닝한 모델"을 내 목소리 테스트셋에서 비교한 CER/WER 표와, 마이크로 말하면 받아쓰는 데모입니다.

```
[Zeroth-Korean 51h] ──► 사전학습 ──► base.pt ──► 파인튜닝 ──► my_voice.pt
                                         ▲
[내 녹음 20~40분] ── train/test 분리 ─────┘ (test는 학습에 사용 안 함)
```

## 결과: 내 목소리로 파인튜닝한 게 어떤 가치가 있었나

DeepSpeech2는 음성을 **글자로 받아쓰는 음성인식 모델**입니다. 그래서 여기서 말하는 "내 목소리 학습"은 내 목소리를 만들어내는 것(음성합성·보이스 클로닝)이 아니라, **내 목소리를 더 잘 알아듣게 만드는 것**입니다.

| 테스트셋 | 사전학습 모델 CER | 내 목소리로 파인튜닝 CER |
|---|---|---|
| 내 목소리 (학습에 안 쓴 19문장) | 0.261 | **0.165** (37% 개선) |
| 다른 사람 목소리 (Zeroth 457문장) | **0.171** | 0.237 (39% 악화) |

- **효과는 실제로 있었습니다.** 같은 모델이 녹음 **7분**(112문장)만으로 내 목소리 CER이 0.261 → 0.165로 줄었습니다. 음성인식에서는 이걸 **화자 적응(speaker adaptation)** 이라고 부르며, 실제로 쓰이는 기법입니다.
  - 예: `냉장고에 우유가 다 떨어졌네요` → 사전학습 `맹장보의 우휴가 다 떨어젼데요` / 파인튜닝 `냉장보의 우유가 다 떨어졌네요`
- **실제로는 이런 곳에 씁니다.** 발음이 독특하거나 사투리가 있는 사람, 구음장애가 있는 사람의 음성인식, 그리고 회사 용어처럼 특정 분야 단어를 잘 알아듣게 할 때입니다.
- **대가도 숫자로 보였습니다.** 다른 사람 목소리는 0.171 → 0.237로 나빠졌습니다. 적은 데이터로 파인튜닝하면 원래 알던 것을 잊는 **망각(catastrophic forgetting)** 이 그대로 나타난 것이고, 학습용으로는 오히려 좋은 관찰 거리입니다.
- **DS2 부품을 한 번에 조립해 본 결과물입니다.** [how-ai-works](https://github.com/dev-jonghoonpark/how-ai-works)에서 따로 다룬 행 합성곱, CTC 빔 서치, RNN 드롭아웃 같은 DS2 구성 요소가 실제 학습 파이프라인 안에서 어떻게 맞물리는지 직접 돌려 볼 수 있습니다.

주의할 점: 파인튜닝 모델은 테스트 19문장에서 CER이 가장 낮았던 epoch을 골랐으므로 0.165는 약간 낙관적인 수치입니다(고르지 않은 마지막 epoch도 0.177). 테스트가 19문장뿐이라 문장 몇 개에 따라 수치가 크게 흔들립니다.

## 구조

```
ds2/
  tokenizer.py   한글 음절 ↔ 자모(초성/중성/종성) 토큰, CTC blank
  features.py    16kHz 로딩, 선형 log 스펙트로그램(20ms/10ms, 161 bin), SpecAugment
  model.py       DeepSpeech2: Conv2d x2 → BiGRU x5 (sequence-wise BN) → Linear
  data.py        JSONL 매니페스트, 패딩 collate, SortaGrad 샘플러
  decode.py      greedy CTC 디코딩, CER/WER
scripts/
  prepare_zeroth.py   공개 데이터 다운로드 (~3GB)
  record.py           문장을 보여주고 내 목소리 녹음
  split_my_voice.py   녹음을 train/test로 분리
  train.py            학습 / 파인튜닝 (--init, --freeze-conv, --freeze-rnn)
  compare.py          체크포인트끼리 CER/WER 비교 (결과물)
  transcribe.py       파일 또는 마이크 받아쓰기
prompts/ko_prompts.txt  녹음용 문장 130개
```

## 준비

```bash
sudo apt install libportaudio2   # 마이크 녹음용 (record.py, transcribe.py --mic)
uv sync
```

RTX 3070(8GB) 기준으로 설정했습니다. 메모리가 부족하면 `--batch-size`나 `--max-duration`을 줄이세요.

## 실행 순서

### 1. 공개 데이터로 사전학습 (가장 오래 걸림)

```bash
uv run scripts/prepare_zeroth.py --out data/zeroth
uv run scripts/train.py --train data/zeroth/train.jsonl --valid data/zeroth/test.jsonl \
    --epochs 30 --lr 3e-4 --out checkpoints/base
```

- 파라미터 약 18.5M. 3070에서 epoch당 대략 10~20분 예상(디스크/CPU에 따라 다름).
- 처음 몇 epoch은 공백이나 빈 문자열만 출력합니다. CTC 모델의 정상적인 초기 현상(blank만 내보내는 단계)이니 기다리세요.
- 흐름만 먼저 확인하려면 `--limit 2000`으로 일부만 받아 짧게 돌려 보세요.
- 51시간은 DS2 기준으로 매우 적은 양(논문은 영어 11,940시간)이라 base 모델 CER은 높게 나옵니다. 이것도 관찰할 거리입니다.

### 2. 내 목소리 녹음

따로 준비할 오디오 파일은 없습니다. `record.py`가 `prompts/ko_prompts.txt`의 문장을 화면에 하나씩 보여주면, **그 문장을 소리 내어 읽기만 하면** 자동으로 저장됩니다.

```bash
uv run scripts/record.py --prompts prompts/ko_prompts.txt --out data/my_voice
uv run scripts/split_my_voice.py --dir data/my_voice
```

녹음 화면은 이렇게 진행됩니다.

```
[3/130] 회의가 삼십 분 늦게 시작한다고 합니다
  Enter=녹음, s=건너뛰기, q=종료 >        ← Enter 누르고 문장 읽기
  ● 녹음 중... 끝나면 Enter               ← 다 읽으면 Enter
  길이 2.8s, 최대 음량 0.42
  저장? (Enter=저장, p=들어보기, r=다시) >
```

- `p`를 누르면 방금 녹음을 들어볼 수 있습니다(여러 번 가능). 잘렸거나 잡음이 크면 `r`로 다시 녹음하세요.
- `⚠ 너무 작음` 또는 `⚠ 클리핑` 경고가 뜨면 `r`로 다시 녹음하세요.
- `s`는 건너뛰기, `q`는 종료입니다. 다시 실행하면 녹음하던 곳부터 이어집니다.
- 마이크가 안 잡히면 `python -m sounddevice`로 장치 번호를 확인하고 `--device 번호`를 붙이세요.
- 조용한 곳에서, 마이크 거리는 일정하게, 또박또박보다는 평소 말투로 읽으세요.
- 결과는 `data/my_voice/wav/*.wav`와 `metadata.jsonl`(파일 경로, 문장, 길이)로 저장됩니다.

**문장 추가하기**

- 130문장이면 10분 남짓입니다. 문장을 추가해 20~40분 정도 모으면 개선이 확실히 보입니다.
- `prompts/ko_prompts.txt`에 한 줄에 한 문장씩 쓰면 됩니다. `#`으로 시작하는 줄은 무시됩니다.
- 받아쓰기에 쓰고 싶은 표현(내 이름, 자주 쓰는 용어 등)을 넣어 두면 좋습니다. 숫자·영어는 한글로 풀어 쓰세요(모델은 한글 음절만 출력). 예: `7시` → `일곱 시`

**train/test 분리**

`split_my_voice.py`는 녹음을 train 85% / test 15%로 나눠 `train.jsonl`, `test.jsonl`을 만듭니다. test 문장은 학습에 쓰지 않으므로 파인튜닝 전후를 공정하게 비교할 수 있습니다.

### 3. 파인튜닝

```bash
uv run scripts/train.py --init checkpoints/base/best.pt \
    --train data/my_voice/train.jsonl --valid data/my_voice/test.jsonl \
    --epochs 40 --lr 5e-5 --batch-size 16 --freeze-conv --out checkpoints/my_voice
```

epoch 0에서 파인튜닝 전 CER을 먼저 출력하므로 바로 비교할 수 있습니다.

### 4. 결과 확인

```bash
uv run scripts/compare.py --test data/my_voice/test.jsonl \
    --ckpt base=checkpoints/base/best.pt mine=checkpoints/my_voice/best.pt

uv run scripts/transcribe.py --ckpt checkpoints/my_voice/best.pt --mic
```

### 5. 데모 페이지 (GitHub Pages)

`docs/`는 빌드 도구 없이 그대로 배포되는 정적 사이트입니다. 테스트 문장마다 오디오를 틀고, 사전학습/파인튜닝 모델의 받아쓰기를 틀린 글자 표시와 함께 비교하며, 학습 곡선도 보여줍니다.

```bash
uv run scripts/build_demo.py              # docs/data.js, docs/audio/ 생성
cd docs && python -m http.server 8000     # http://localhost:8000 에서 미리보기
```

- `build_demo.py`는 학습 로그에서 학습 곡선을 읽습니다. 기본 경로는 `checkpoints/base/train.log`, `checkpoints/my_voice.log`이므로 학습할 때 `2>&1 | tee checkpoints/my_voice.log`처럼 로그를 남겨 두세요 (`--base-log`, `--mine-log`로 바꿀 수 있음).
- 샘플은 내 목소리 테스트 문장 전부와, Zeroth 테스트셋 앞쪽의 10초 이하 문장 8개입니다(`--zeroth-samples`).
- `index.html`, `app.js`, `style.css`는 직접 수정하는 파일이고, `data.js`와 `audio/`는 스크립트가 다시 만듭니다.

배포: 저장소에 `docs/`까지 커밋하고 push → GitHub 저장소 **Settings → Pages → Build and deployment**에서 Source를 `Deploy from a branch`, 브랜치 `main`, 폴더 `/docs`로 지정합니다.

> ⚠️ `docs/audio/`에는 **내 목소리 녹음 원본**이 들어갑니다. GitHub Pages는 공개 사이트입니다. 올리기 전에 녹음 내용을 확인하세요. 목소리를 빼고 싶으면 `--my-samples 0`으로 빌드하면 수치와 곡선만 남습니다.

## 해 볼 만한 실험

| 실험 | 방법 | 관찰 포인트 |
|---|---|---|
| 사전학습 없이 내 목소리만 | `--init` 없이 my_voice로 학습 | train loss는 떨어지는데 test CER은 안 떨어짐 = 과적합 |
| 어떤 층을 고정할까 | `--freeze-conv`, `--freeze-rnn 2/4` | 데이터가 적을수록 많이 고정하는 게 유리한가 |
| 학습률 | `--lr 1e-5 / 5e-5 / 3e-4` | 너무 크면 base가 가진 일반 지식을 잊음(catastrophic forgetting) |
| 망각 측정 | 파인튜닝 모델로 `compare.py --test data/zeroth/test.jsonl` | 내 목소리는 좋아지고 다른 사람 목소리는 나빠지는가 |
| 데이터 양 | 녹음 5/10/20분으로 나눠 파인튜닝 | 몇 분부터 효과가 나는가 |
| SpecAugment | `--no-augment` | 적은 데이터에서 증강의 효과 |
| SortaGrad | `data.py`에서 첫 epoch 정렬 제거 | 초기 학습 안정성 |

## 설계 메모

- **왜 자모 단위인가**: 음절은 11,172종이라 출력층이 크고 데이터가 드문 음절은 학습이 안 됩니다. 초성/중성/종성 67개 + 공백으로 분해하고 디코딩 후 다시 조합합니다. 초성 ㄱ과 종성 ㄱ은 다른 토큰이라 조합이 모호하지 않습니다.
- **왜 CER인가**: 한국어는 띄어쓰기가 흔들려서 WER이 과하게 나쁘게 나옵니다. CER은 공백을 빼고 계산합니다.
- **논문과 다른 점**: 모델 크기 축소, 양방향 GRU 출력을 concat 대신 합산, AdamW + warmup/cosine (논문은 SGD+Nesterov), SpecAugment 추가, 언어모델 beam search 없음(greedy).
- **파인튜닝 시 BN**: 고정한 층은 `eval()`로 두어 BatchNorm 통계가 적은 데이터에 흔들리지 않게 합니다.

## 참고: 요즘은 어떻게 하나

DS2(2015)는 학습 목적에 좋은 구조지만 지금 기준으로는 오래된 방식입니다. 실사용 성능이 목표라면 다음을 참고하세요. (이 프로젝트에는 적용하지 않았습니다.)

- **언어모델 결합**: KenLM n-gram + CTC beam search만 붙여도 CER이 크게 줄어듭니다. DS2 논문의 원래 구성입니다.
- **Conformer / Squeezeformer**: RNN 대신 Convolution + Self-attention. CTC 또는 RNN-T와 결합해 DS2보다 훨씬 좋은 성능을 냅니다.
- **자기지도 사전학습 (wav2vec 2.0, HuBERT, XLS-R)**: 라벨 없는 대량 음성으로 사전학습 → 적은 라벨로 파인튜닝. "내 목소리 몇 분"이라는 상황에 가장 잘 맞습니다.
- **Whisper 계열**: 68만 시간 다국어 데이터의 encoder-decoder. 파인튜닝 없이도 한국어가 잘 되고, LoRA로 내 목소리/도메인에 맞출 수 있습니다.
- **화자 적응 기법**: i-vector/x-vector 입력, LHUC처럼 화자별 소수 파라미터만 학습하는 방식.

## 라이선스 / 출처

- Zeroth-Korean: CC BY 4.0 ([kresnik/zeroth_korean](https://huggingface.co/datasets/kresnik/zeroth_korean), 원본 OpenSLR SLR40)
- DeepSpeech2: Amodei et al., "Deep Speech 2: End-to-End Speech Recognition in English and Mandarin", 2015
