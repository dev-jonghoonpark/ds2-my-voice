window.DEMO_DATA = {
 "model": {
  "n_classes": 69,
  "rnn_hidden": 512,
  "rnn_layers": 5,
  "conv_channels": 32,
  "params_m": 18.5
 },
 "train": {
  "zeroth": {
   "count": 22263,
   "minutes": 3100.8
  },
  "my_voice": {
   "count": 112,
   "minutes": 7.4
  }
 },
 "curves": {
  "base": [
   {
    "epoch": 1,
    "cer": 0.776
   },
   {
    "epoch": 2,
    "cer": 0.555
   },
   {
    "epoch": 3,
    "cer": 0.471
   },
   {
    "epoch": 4,
    "cer": 0.439
   },
   {
    "epoch": 5,
    "cer": 0.391
   },
   {
    "epoch": 6,
    "cer": 0.369
   },
   {
    "epoch": 7,
    "cer": 0.325
   },
   {
    "epoch": 8,
    "cer": 0.311
   },
   {
    "epoch": 9,
    "cer": 0.303
   },
   {
    "epoch": 10,
    "cer": 0.29
   },
   {
    "epoch": 11,
    "cer": 0.267
   },
   {
    "epoch": 12,
    "cer": 0.247
   },
   {
    "epoch": 13,
    "cer": 0.244
   },
   {
    "epoch": 14,
    "cer": 0.247
   },
   {
    "epoch": 15,
    "cer": 0.234
   },
   {
    "epoch": 16,
    "cer": 0.216
   },
   {
    "epoch": 17,
    "cer": 0.212
   },
   {
    "epoch": 18,
    "cer": 0.207
   },
   {
    "epoch": 19,
    "cer": 0.199
   },
   {
    "epoch": 20,
    "cer": 0.194
   },
   {
    "epoch": 21,
    "cer": 0.182
   },
   {
    "epoch": 22,
    "cer": 0.186
   },
   {
    "epoch": 23,
    "cer": 0.18
   },
   {
    "epoch": 24,
    "cer": 0.178
   },
   {
    "epoch": 25,
    "cer": 0.178
   },
   {
    "epoch": 26,
    "cer": 0.176
   },
   {
    "epoch": 27,
    "cer": 0.174
   },
   {
    "epoch": 28,
    "cer": 0.173
   },
   {
    "epoch": 29,
    "cer": 0.171
   },
   {
    "epoch": 30,
    "cer": 0.172
   }
  ],
  "mine": [
   {
    "epoch": 0,
    "cer": 0.261
   },
   {
    "epoch": 1,
    "cer": 0.261
   },
   {
    "epoch": 2,
    "cer": 0.232
   },
   {
    "epoch": 3,
    "cer": 0.235
   },
   {
    "epoch": 4,
    "cer": 0.235
   },
   {
    "epoch": 5,
    "cer": 0.219
   },
   {
    "epoch": 6,
    "cer": 0.194
   },
   {
    "epoch": 7,
    "cer": 0.19
   },
   {
    "epoch": 8,
    "cer": 0.19
   },
   {
    "epoch": 9,
    "cer": 0.194
   },
   {
    "epoch": 10,
    "cer": 0.19
   },
   {
    "epoch": 11,
    "cer": 0.19
   },
   {
    "epoch": 12,
    "cer": 0.181
   },
   {
    "epoch": 13,
    "cer": 0.174
   },
   {
    "epoch": 14,
    "cer": 0.174
   },
   {
    "epoch": 15,
    "cer": 0.171
   },
   {
    "epoch": 16,
    "cer": 0.168
   },
   {
    "epoch": 17,
    "cer": 0.168
   },
   {
    "epoch": 18,
    "cer": 0.168
   },
   {
    "epoch": 19,
    "cer": 0.165
   },
   {
    "epoch": 20,
    "cer": 0.171
   },
   {
    "epoch": 21,
    "cer": 0.168
   },
   {
    "epoch": 22,
    "cer": 0.168
   },
   {
    "epoch": 23,
    "cer": 0.168
   },
   {
    "epoch": 24,
    "cer": 0.171
   },
   {
    "epoch": 25,
    "cer": 0.168
   },
   {
    "epoch": 26,
    "cer": 0.171
   },
   {
    "epoch": 27,
    "cer": 0.181
   },
   {
    "epoch": 28,
    "cer": 0.181
   },
   {
    "epoch": 29,
    "cer": 0.177
   },
   {
    "epoch": 30,
    "cer": 0.174
   },
   {
    "epoch": 31,
    "cer": 0.174
   },
   {
    "epoch": 32,
    "cer": 0.174
   },
   {
    "epoch": 33,
    "cer": 0.177
   },
   {
    "epoch": 34,
    "cer": 0.177
   },
   {
    "epoch": 35,
    "cer": 0.177
   },
   {
    "epoch": 36,
    "cer": 0.181
   },
   {
    "epoch": 37,
    "cer": 0.181
   },
   {
    "epoch": 38,
    "cer": 0.177
   },
   {
    "epoch": 39,
    "cer": 0.181
   },
   {
    "epoch": 40,
    "cer": 0.177
   }
  ]
 },
 "sets": {
  "my_voice": {
   "count": 19,
   "metrics": {
    "base": {
     "cer": 0.2613,
     "wer": 0.56
    },
    "mine": {
     "cer": 0.1645,
     "wer": 0.41
    }
   },
   "samples": [
    {
     "audio": "audio/my_voice_00.wav",
     "duration": 3.81,
     "ref": "감기 기운이 있어서 병원에 다녀왔습니다",
     "base": "감기 띠운이 있어서 병원엘 반혀왔습니다",
     "mine": "감기 띠운이 있어서 병원에 다녀왔습니다"
    },
    {
     "audio": "audio/my_voice_01.wav",
     "duration": 3.84,
     "ref": "회의가 삼십 분 늦게 시작한다고 합니다",
     "base": "헤이이가 삼십 군 느께 시작한 가고 합니다",
     "mine": "헤외이가 삼십 분 느께 시작한다고 합니다"
    },
    {
     "audio": "audio/my_voice_02.wav",
     "duration": 4.67,
     "ref": "테스트가 전부 통과하면 배포를 진행하겠습니다",
     "base": "테스트가 전부 통과하면 배포를 진행하겠습니다",
     "mine": "테스트가 점부 통과하면 배포를 진행하겠습니다"
    },
    {
     "audio": "audio/my_voice_03.wav",
     "duration": 4.22,
     "ref": "평소에 말하는 속도로 자연스럽게 읽어 보세요",
     "base": "황소 말하는 속도로 자연스럽게 입어보세요",
     "mine": "퐝소 말하는 속도로 자연 스럽게 읽어보세요"
    },
    {
     "audio": "audio/my_voice_04.wav",
     "duration": 3.81,
     "ref": "코드 리뷰 의견을 반영해서 다시 올렸어요",
     "base": "코드리이 의견으 반영해서 다시 올렸어요",
     "mine": "코드리비 의견을 반양해서 다시 올렸어요"
    },
    {
     "audio": "audio/my_voice_05.wav",
     "duration": 3.46,
     "ref": "비행기가 한 시간 정도 지연되었습니다",
     "base": "비행기가 한 시간 정도 지연되었습니다",
     "mine": "비행기가 한 시간 점도 지연되었습니다"
    },
    {
     "audio": "audio/my_voice_06.wav",
     "duration": 2.98,
     "ref": "오늘 하루도 정말 수고 많으셨습니다",
     "base": "오늘 하루도 정말 수고 많으셨습니다",
     "mine": "오늘 하루도 정말 쑤고 많으셨습니다"
    },
    {
     "audio": "audio/my_voice_07.wav",
     "duration": 3.74,
     "ref": "그럼 다음 주에 다시 뵙겠습니다",
     "base": "그런 바음주에 다시 뵑괬습니다",
     "mine": "그런 다음 주에 다시 뵑괬습니다"
    },
    {
     "audio": "audio/my_voice_08.wav",
     "duration": 3.46,
     "ref": "불꽃놀이가 밤 아홉 시에 시작한대요",
     "base": "붉근 노리가 파마오시에 시작한데",
     "mine": "부근 노리가파 마오 씨에 시작한데요"
    },
    {
     "audio": "audio/my_voice_09.wav",
     "duration": 4.64,
     "ref": "고속도로는 귀성 차량으로 크게 붐비고 있습니다",
     "base": "고속도로는 기성 차략으로 크게 붕되고 있습니다",
     "mine": "고속도로는 기성 차량으로 크게 분비고 있습니다"
    },
    {
     "audio": "audio/my_voice_10.wav",
     "duration": 3.55,
     "ref": "팀장님께 휴가 신청을 드렸습니다",
     "base": "김정님께 수가 신사어 들였습니다",
     "mine": "김장님께 휴가 신처을 드렸습니다"
    },
    {
     "audio": "audio/my_voice_11.wav",
     "duration": 4.64,
     "ref": "저녁에는 된장찌개랑 계란말이를 만들 거예요",
     "base": "전여에는 댄당찌 계랑 개란마리를 만들 거예",
     "mine": "전여게는 댄장찌개랑 개란마리를 만들 거예요"
    },
    {
     "audio": "audio/my_voice_12.wav",
     "duration": 4.26,
     "ref": "고양이가 키보드 위에 올라가서 잠들었어요",
     "base": "고양이가 키보들 위에 올라가서 잠드렀어요",
     "mine": "고양이가 키보들 위에 올라가서 잠드렀어요"
    },
    {
     "audio": "audio/my_voice_13.wav",
     "duration": 3.68,
     "ref": "시험 공부 때문에 요즘 잠을 많이 못 잤어요",
     "base": "시엄 공보 때문에 유즘 잠을 많이 못닸어요",
     "mine": "시엄 공보 때문에 요즘 잠을 많이 못 잤어요"
    },
    {
     "audio": "audio/my_voice_14.wav",
     "duration": 3.39,
     "ref": "냉장고에 우유가 다 떨어졌네요",
     "base": "맹장보의 우휴가 다 떨어젼데요",
     "mine": "냉장보의 우유가 다 떨어졌네요"
    },
    {
     "audio": "audio/my_voice_15.wav",
     "duration": 4.29,
     "ref": "화면 공유를 시작하겠습니다 잘 보이시나요",
     "base": "하면 공율에이 시작하게 슨거가 자 보있스나요",
     "mine": "하면 공율를 시작하겠습니다 잘 보이 시나요"
    },
    {
     "audio": "audio/my_voice_16.wav",
     "duration": 3.39,
     "ref": "눈이 많이 와서 길이 미끄러워요",
     "base": "눈이 많이 와서 길이 미끌어고요",
     "mine": "눈이 많이 와서 길이 미끌어워요"
    },
    {
     "audio": "audio/my_voice_17.wav",
     "duration": 4.03,
     "ref": "지역 축제에 많은 관광객이 몰렸습니다",
     "base": "지역 속드에 많흔 관광객이 몰렸습니다",
     "mine": "지역 숙제에 많은 관광객이 몰렸습니다"
    },
    {
     "audio": "audio/my_voice_18.wav",
     "duration": 3.14,
     "ref": "좋은 의견 주셔서 정말 감사합니다",
     "base": "조은 의견 주에서 정말 감사합니다",
     "mine": "좋은으견 줄에서 정말 감사합니다"
    }
   ]
  },
  "zeroth": {
   "count": 457,
   "metrics": {
    "base": {
     "cer": 0.1711,
     "wer": 0.4201
    },
    "mine": {
     "cer": 0.237,
     "wer": 0.5629
    }
   },
   "samples": [
    {
     "audio": "audio/zeroth_00.wav",
     "duration": 8.54,
     "ref": "지난해 이들 크루즈관광객의 평균 체류기간은 오 쩜 구 사 시간",
     "base": "지난해 이들 크루즈 강강객의 평균 체류기간일 구점 구 사 시간",
     "mine": "지난해 이들 크루즈 강낭객의 평균 채료 기강을 주 쩜 구사 시간에"
    },
    {
     "audio": "audio/zeroth_01.wav",
     "duration": 8.16,
     "ref": "평소 오전 아홉 시 에서 오후 일곱 시까지 일하면 하루 이 만원 정도를 번다",
     "base": "편소 오전 아홉 시에서 오후 일곱 히깔지 일하면 하루 이 만원 정도를 번다",
     "mine": "팸소 오전 아홉 시에서 오후 일곱 식깔지 일하믄 하룬 이 만 원 정도를 반다"
    },
    {
     "audio": "audio/zeroth_02.wav",
     "duration": 8.93,
     "ref": "야권은 여당에서 분명한 입장을 표해야 한다고 연일 압박 수위를 높이고 있다",
     "base": "야권은 여당에서 분명한 입장을 표해야 한다고 명일 압박수위를 옾이고 있다",
     "mine": "야권은 여장이서 분명한 입장을 표해야 한다고 영일 압박 수위를 옾이고 있다"
    },
    {
     "audio": "audio/zeroth_03.wav",
     "duration": 9.22,
     "ref": "젠슨 황은 엔비디아에서 매출의 삼십 퍼센트 이상을 연구개발에 투자한다",
     "base": "제슨 황은 엔비기아이다 매출의 삼십 퍼센트 이상을 연구개발을 툐자한다",
     "mine": "젤신 황은 엔비기아 에사 매출이 삼십 퍼센트 이상을 연구개발를 투자핬다"
    },
    {
     "audio": "audio/zeroth_04.wav",
     "duration": 8.51,
     "ref": "몽둥이와 함께 변 씨의 가게에선 쇠 파이프와 곡괭이 자루까지 발견됐습니다",
     "base": "몽둥이와 함께 변씨이 갈게에살 새파프와 곡괜이 자루깔지 발견됐습니다",
     "mine": "몽등이와 함께 변 씨이 갈게 에설 새파프와 곡괭이 자루까지 발견됐습니다"
    },
    {
     "audio": "audio/zeroth_05.wav",
     "duration": 6.95,
     "ref": "아울러 미약하게나마 제가 할 수 있는 일을 찾아 실천하겠습니다",
     "base": "아울러 미약하게 나마 제가 할 수 있는 일을 찾아 실천하겠습니다",
     "mine": "아울러 미약하게나마 제가 할 수 있는 일을 찾아 실찬하겠습니다"
    },
    {
     "audio": "audio/zeroth_06.wav",
     "duration": 9.18,
     "ref": "이미 알려진 대로 미군은 한반도에 세 개 사드 포대 기지 배치가 필요하다는 입장이다",
     "base": "이미 알려진 대로 미군은 한반도에 세개 사데 포대 기지의 배치가 필요할대는 입장이다",
     "mine": "이미 알려진 대로 미군은 한 반도에 세게 싸된 포드 기지에 데치가 필요 할다는 입장이다"
    },
    {
     "audio": "audio/zeroth_07.wav",
     "duration": 7.63,
     "ref": "손재주가 별로 없는 사람들은 처음에 자주 상처를 입었을 것입니다",
     "base": "손재주가 별로 없는 사람들은 처음에 자주 산서를 입었을 것입니다",
     "mine": "순재 주가 별로 없는 사람들은 차음에 잘 주 산사를 입었을 것입니다"
    }
   ]
  }
 }
};
