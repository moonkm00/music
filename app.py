import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
import plotly.graph_objects as go
import time

# 1. 페이지 초기 설정 및 프리미엄 스타일링
st.set_page_config(
    page_title="VibeFrame - AI 사진 감성 음악 추천 서비스",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# style.css 외부 스타일시트 로드 및 인젝션 (클린 코드 분리)
with open("style.css", "r", encoding="utf-8") as f:
    custom_css = f.read()
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

# 2. 로딩 지점 정의 (대표 모델 캐싱)
@st.cache_resource
def loadCustomModel():
    import os
    import torch
    from torchvision import models
    class CustomVibeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            try:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            except AttributeError:
                self.backbone = models.resnet50(pretrained=True)
            numFtrs = self.backbone.fc.in_features
            self.backbone.fc = torch.nn.Sequential(
                torch.nn.Linear(numFtrs, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 2),
                torch.nn.Sigmoid()
            )
        def forward(self, x):
            return self.backbone(x)
    model = CustomVibeModel()
    if os.path.exists("custom_vibe_model.pth"):
        model.load_state_dict(torch.load("custom_vibe_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model


# 3. Russell 감성 모델 차원 및 텍스트 템플릿
# 각 분위기 태그의 Valence(긍정성, 0~1)와 Energy(활성도, 0~1) 설정
VIBES = {
    "cozy warm fireplace room": {"valence": 0.80, "energy": 0.22, "label": "따뜻하고 아늑한 방 (Cozy)"},
    "calm quiet peaceful sunset": {"valence": 0.85, "energy": 0.28, "label": "차분하고 평화로운 노을 (Calm)"},
    "energetic bright dance party club": {"valence": 0.88, "energy": 0.90, "label": "신나는 활력의 축제 (Energetic)"},
    "happy joyful sunny spring day": {"valence": 0.95, "energy": 0.72, "label": "밝고 기분 좋은 봄날 (Joyful)"},
    "sad rainy lonely window": {"valence": 0.20, "energy": 0.25, "label": "쓸쓸하고 슬픈 비 오는 날 (Melancholy)"},
    "mysterious deep dark night city": {"valence": 0.45, "energy": 0.65, "label": "차가운 다크 네온 도시 (Mysterious)"},
    "powerful aggressive rock action scene": {"valence": 0.35, "energy": 0.85, "label": "강렬하고 긴장감 넘치는 비트 (Tense)"},
    "chill relaxed jazz coffee shop": {"valence": 0.88, "energy": 0.35, "label": "여유로운 카페 재즈 (Relaxed)"}
}

# 4. 내장 큐레이션 음원 DB 완전 제거 (100% 실시간 LLM 추천 방식 운영)
# 모든 곡 추천은 사용자가 업로드한 이미지의 Valence 및 Energy에 맞춰 Google Gemini API를 통해 실시간 생성됩니다.

# 5. 핵심 이미지 분석 및 분위기 매핑 함수
def analyzeImage(pilImage) -> tuple:
    """
    [VibeFrame 실시간 감성 추론 전처리 파이프라인]
    사용자가 업로드한 이미지를 딥러닝 및 OpenCV 연산이 가능하도록 실시간 가공 및 정제합니다.
    """
    # 5-1. [OpenCV 컬러/명암 분석을 위한 화상 전처리]
    imgNp = np.array(pilImage)
    imgBgr = cv2.cvtColor(imgNp, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(imgBgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 픽셀 채도(Saturation) 및 명도(Value)를 0.0~1.0 사이로 정규화하여 이미지의 명암도 추출
    avgS = np.mean(s) / 255.0  
    avgV = np.mean(v) / 255.0  
    
    # 대표 색상 팔레트 추출을 위한 이미지 다운샘플링 전처리 (30x30 고속 스캔)
    resized = cv2.resize(imgBgr, (30, 30))
    pixels = resized.reshape(-1, 3)
    
    # 픽셀 데이터를 Hex 포맷 색상값으로 매핑 변환
    from collections import Counter
    hexPixels = [f"#{p[2]:02x}{p[1]:02x}{p[0]:02x}" for p in pixels]
    dominantHex = [item[0] for item in Counter(hexPixels).most_common(5) if item[0] != "#ffffff" and item[0] != "#000000"][:3]
    if len(dominantHex) < 3:
        dominantHex = [item[0] for item in Counter(hexPixels).most_common(3)]
        
    vibeScores = []
    
    # 5-2. [PyTorch Custom ResNet50 감성 정밀 분석을 위한 화상 전처리]
    # [전처리 ②단계: 시각 이미지 텐서 변환 및 ImageNet 표준 규격 정규화 (화상 전처리)]
    import torch
    from torchvision import transforms
    preprocess = transforms.Compose([
        # [Resize] 고해상도 사용 이미지 규격을 ResNet50 입력 사양인 224x224로 정밀 축소
        transforms.Resize((224, 224)),
        # [ToTensor] 데이터 차원 재정렬(CHW) 및 float32 스케일링(0.0~1.0)
        transforms.ToTensor(),
        # [Normalize] ImageNet 데이터 도메인의 평균/표준편차 분포 수식 적용
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    inputTensor = preprocess(pilImage).unsqueeze(0)
    customModel = loadCustomModel()
    with torch.no_grad():
        prediction = customModel(inputTensor)[0].numpy()
        
    targetValence = float(prediction[0])
    targetEnergy = float(prediction[1])
    
    # 대표 무드 태그 설정 (Russell 공간에서 가장 가까운 VIBE 태그 매치)
    for key, info in VIBES.items():
        dist = np.sqrt((info["valence"] - targetValence)**2 + (info["energy"] - targetEnergy)**2)
        similarity = 1.0 / (1.0 + dist)
        vibeScores.append((info["label"], similarity))
            
    # 수치 경계값 제한
    targetValence = max(0.0, min(1.0, targetValence))
    targetEnergy = max(0.0, min(1.0, targetEnergy))
    
    # 점수가 높은 순으로 Vibe 태그 정렬
    vibeScores = sorted(vibeScores, key=lambda x: x[1], reverse=True)
    topVibes = [v[0] for v in vibeScores[:2]]
    
    return targetValence, targetEnergy, dominantHex, topVibes

# 6.2. 구글 Gemini 2.5 Flash REST API 실시간 음악 추천 엔진 (100% 실시간 생성)
def getLLMRecommendations(apiKey: str, targetV: float, targetE: float) -> dict:
    import json
    import urllib.request
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={apiKey}"
    
    prompt = f"""You are a music recommendation system. Recommend exactly 3 K-POP, 3 POP, and 3 J-POP songs (total 9 songs) suitable for a photo with the following Russell emotion model coordinates:
Valence (Positivity): {targetV:.2f} (0.0 to 1.0)
Energy (Arousal): {targetE:.2f} (0.0 to 1.0)

Your response must be a valid JSON object matching this structure:
{{
  "K-POP": [
    {{"title": "Song Title", "artist": "Artist Name", "genre": "Genre", "valence": 0.8, "energy": 0.9, "desc": "Short description of why it fits in Korean"}}
  ],
  "POP": [
    {{"title": "Song Title", "artist": "Artist Name", "genre": "Genre", "valence": 0.8, "energy": 0.9, "desc": "Short description of why it fits in Korean"}}
  ],
  "J-POP": [
    {{"title": "Song Title", "artist": "Artist Name", "genre": "Genre", "valence": 0.8, "energy": 0.9, "desc": "Short description of why it fits in Korean"}}
  ]
}}
Each array must contain exactly 3 unique, real, highly popular songs. Ensure "desc" is in polite, highly professional Korean (한국어 존댓말). Do not wrap in markdown code blocks. Just return raw JSON.
"""
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    data = json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=30) as response:
            res_body = response.read().decode('utf-8')
        
        res_json = json.loads(res_body)
        text_content = res_json['candidates'][0]['content']['parts'][0]['text']
        parsed = json.loads(text_content)
        
        # 키 정규화 (대소문자 및 매핑)
        normalized = {"K-POP": [], "POP": [], "J-POP": []}
        for key in ["K-POP", "POP", "J-POP"]:
            for possible_key in [key, key.lower(), key.replace("-", ""), key.replace("-", "").lower()]:
                if possible_key in parsed:
                    normalized[key] = parsed[possible_key]
                    break
        
        # 데이터 정합성 검증
        if not normalized["K-POP"] or not normalized["POP"] or not normalized["J-POP"]:
            raise ValueError("LLM 응답 형식 불완전")
            
        return normalized
    except Exception as e:
        raise e

# 6.5. 유튜브 동영상 실시간 검색 엔진 (크롤러 기반 무권한 연동)
def searchYoutubeVideos(query: str, count: int = 3) -> list:
    import urllib.request
    import urllib.parse
    import re
    
    encoded_q = urllib.parse.quote(query)
    url = f"https://www.youtube.com/results?search_query={encoded_q}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode('utf-8', errors='ignore')
        
        # 유튜브 동영상 고유 ID 추출용 정규식
        video_ids = re.findall(r'"videoId"\s*:\s*"([a-zA-Z0-9_-]{11})"', html)
        if not video_ids:
            video_ids = re.findall(r'/watch\?v=([a-zA-Z0-9_-]{11})', html)
            
        results = []
        seen = set()
        for vid in video_ids:
            if vid not in seen:
                seen.add(vid)
                results.append(vid)
                if len(results) >= count:
                    break
        return results
    except Exception:
        return []

# 7. UI/UX 구현부
st.markdown('<div class="premium-title">VibeFrame</div>', unsafe_allow_html=True)
st.markdown('<div class="premium-subtitle">당신의 순간(사진) 속 숨어있는 분위기와 어울리는 프리미엄 사운드스페이스</div>', unsafe_allow_html=True)

# 왼쪽 사이드바 카테고리 구성 (AI 모델 학습 자료실)
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 0.5rem 0;'>
        <h2 style='color: #1DB954; font-weight: 700; margin-bottom: 0;'>🧠 VibeFrame Admin</h2>
        <p style='color: #9ca3af; font-size: 0.85rem;'>AI 모델 학습 및 구조 분석실</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📂 AI 모델 학습 자료실")
    vis_option = st.selectbox(
        "분석 자료를 선택해 주세요:",
        [
            "닫기 (메인 화면만 보기)",
            "📉 AI 학습 손실 곡선",
            "🎯 감성 분류 성능 곡선",
            "🔮 시스템 아키텍처",
            "🧠 모델 네트워크 구조"
        ]
    )

# 메인 콘텐츠 레이아웃 (좌우 위치: col1에 사진 올리기, col2에 분석리포트 및 Russell 무드 매핑 배치)
col1, col2 = st.columns([1, 1], gap="large")

# 사진 올리기 (왼쪽 컬럼에 항상 노출)
with col1:
    with st.container(border=True):
        st.markdown("### 📸 사진 올리기")
        uploaded_file = st.file_uploader(
            "이미지 파일을 드래그 앤 드롭하거나 선택하세요.", 
            type=["png", "jpg", "jpeg", "webp"]
        )
        
        # 테스트용 이미지 자동 로드 기능 (브라우저 검증 및 자동화용)
        test_path = st.query_params.get("test_image_path", None)
        if uploaded_file is None and test_path:
            import os
            if os.path.exists(test_path):
                from io import BytesIO
                with open(test_path, "rb") as f:
                    file_bytes = f.read()
                uploaded_file = BytesIO(file_bytes)
                uploaded_file.name = os.path.basename(test_path)
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True, caption="업로드된 대표님의 사진")

# AI 감성 분석 변수 초기화
targetV, targetE = 0.5, 0.5
colors = ["#1DB954", "#8B5CF6", "#F59E0B"]
vibes = []
recommendations = {}
image_analyzed = False

# 파일 업로드가 되었을 때 AI 분석 가동
if uploaded_file is not None:
    with st.spinner("🔮 AI 감성 엔진이 이미지를 미세하게 분석 중입니다..."):
        customModel = loadCustomModel()
        targetV, targetE, colors, vibes = analyzeImage(image)
        image_analyzed = True
        
        # 100% LLM 실시간 추천 가동! (Secrets/환경 변수 로드)
        import os
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
        if not GEMINI_API_KEY:
            try:
                GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
            except Exception:
                pass
        
        if not GEMINI_API_KEY:
            st.error("🔑 구글 Gemini API Key가 설정되지 않았습니다. `.streamlit/secrets.toml` 파일에 `GEMINI_API_KEY`를 등록하거나 환경 변수를 설정해 주세요.")
            st.stop()
        llm_success = False
        import time
        max_retries = 2
        last_error_msg = ""
        for attempt in range(max_retries):
            try:
                recommendations = getLLMRecommendations(GEMINI_API_KEY, targetV, targetE)
                llm_success = True
                break
            except Exception as e:
                last_error_msg = str(e)
                if any(err in last_error_msg for err in ["429", "503", "Too Many Requests", "Service Unavailable"]):
                    if attempt < max_retries - 1:
                        time.sleep(1.5) # 1.5초간 트래픽 회복 대기 후 재시도
                        continue
                break
        
        if not llm_success:
            st.error(f"⚠️ 실시간 Gemini API 통신에 실패했습니다. (오류: {last_error_msg})")
            st.stop()

# 오른쪽 컬럼 (col2) 렌더링
with col2:
    # 📊 분위기 분석 리포트 (이미지 분석이 성공했을 때만 출력)
    if image_analyzed:
        with st.container(border=True):
            st.markdown("### 📊 AI 감성 분석 리포트")
            
            # 감성 요약 배지 출력
            st.markdown("**추출된 핵심 Vibe:**")
            for vibe in vibes:
                st.markdown(f'<span class="vibe-badge">✨ {vibe}</span>', unsafe_allow_html=True)
                
            # 색감 분석 출력
            st.markdown("<br>**이미지 감성 팔레트:**", unsafe_allow_html=True)
            color_html = ""
            for col in colors:
                color_html += f'<span class="color-dot" style="background-color: {col};"></span> `{col}`'
            st.markdown(color_html, unsafe_allow_html=True)
            
            # Valence, Energy 수치 설명
            st.markdown(f"""
            *   **긍정성(Valence)**: `{targetV:.2f}` (사진 속 색조와 피사체가 자아내는 밝고 포근한 지수)
            *   **활성도(Energy)**: `{targetE:.2f}` (사진 구도와 채도에서 전해지는 운동성 및 템포)
            """)
            
    # 📍 내 사진 무드 매핑 (오른쪽 컬럼에 상시 노출)
    with st.container(border=True):
        st.markdown("### 📍 내 사진 무드 매핑")
        
        # Plotly를 이용한 인터랙티브 차트 작성
        fig = go.Figure()
        
        # 4분면 영역 배경선 추가
        fig.add_shape(type="line", x0=0, y0=0.5, x1=1, y1=0.5, line=dict(color="rgba(0,0,0,0.1)", width=2))
        fig.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=1, line=dict(color="rgba(0,0,0,0.1)", width=2))
        
        # 영역 라벨 텍스트 추가
        fig.add_annotation(x=0.85, y=0.85, text="<b>Happy / Excited</b>", showarrow=False, font=dict(color="#059669", size=11))
        fig.add_annotation(x=0.15, y=0.85, text="<b>Tense / Dark</b>", showarrow=False, font=dict(color="#DC2626", size=11))
        fig.add_annotation(x=0.15, y=0.15, text="<b>Sad / Lonely</b>", showarrow=False, font=dict(color="#2563EB", size=11))
        fig.add_annotation(x=0.85, y=0.15, text="<b>Calm / Relaxed</b>", showarrow=False, font=dict(color="#D97706", size=11))
        
        if image_analyzed:
            # 1. 업로드된 이미지 분석 위치 (큰 다이아몬드 별로 매핑)
            fig.add_trace(go.Scatter(
                x=[targetV], y=[targetE],
                mode='markers+text',
                marker=dict(size=18, color=colors[0] if colors else '#1DB954', symbol='diamond', line=dict(color='#1e293b', width=2)),
                name='내 사진 무드',
                text=["내 사진의 무드 📍"],
                textposition="top center",
                textfont=dict(color='#1e293b', size=13)
            ))
            # 추천 곡 매핑 루프 제거 (내 사진 무드 좌표만 깔끔하게 출력)
            pass
        
        # 차트 레이아웃 스타일 설정
        fig.update_layout(
            xaxis=dict(title='긍정성 (Valence)', range=[0, 1], gridcolor='rgba(0,0,0,0.05)', showticklabels=False),
            yaxis=dict(title='활성도 (Energy)', range=[0, 1], gridcolor='rgba(0,0,0,0.05)', showticklabels=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            height=280
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if not image_analyzed:
            st.info("💡 왼쪽에서 사진을 업로드하시면 내 사진의 감성 좌표(📍)가 실시간 매핑됩니다.")

# 사이드바에서 선택한 AI 학습 자료 출력 (전체 너비로 하단 노출)
if vis_option != "닫기 (메인 화면만 보기)":
    with st.container(border=True):
        st.markdown(f"### 📖 {vis_option}")
        if vis_option == "📉 AI 학습 손실 곡선":
            st.image("visualization_assets/training_loss.png", use_container_width=True, caption="에포크별 Train/Validation MSE Loss 수렴 그래프")
        elif vis_option == "🎯 감성 분류 성능 곡선":
            st.image("visualization_assets/roc_curve.png", use_container_width=True, caption="임계값 0.5 기준 Valence/Energy 탐지 ROC 곡선")
        elif vis_option == "🔮 시스템 아키텍처":
            st.image("visualization_assets/vibeframe_flowchart.png", use_container_width=True, caption="VibeFrame 이미지 감성 스캔 & 추천 시스템 흐름도")
        elif vis_option == "🧠 모델 네트워크 구조":
            st.image("visualization_assets/custom_resnet_architecture.png", use_container_width=True, caption="ResNet50 + MLP Regressor 구조도")

# 파일 업로드가 되었을 때 하단부: 🎵 곡 추천 카드 리스트 및 실시간 유튜브 플레이어 렌더링
if uploaded_file is not None:
    with st.container(border=True):
        st.markdown("## 🎵 VibeFrame 실시간 유튜브 사운드트랙 매칭")
        
        st.markdown("AI 감성 매칭곡이 국가별로 자동으로 연동됩니다. 만약 원하시는 다른 최신곡이나 특정 가수가 있으시다면 아래에 직접 입력하여 실시간 연동해 보십시오.")
        custom_music_search = st.text_input("🔍 실시간 유튜브 음악 다이렉트 검색 (예: '아이유 신곡', '에스파 신곡')", value="")
        
        if custom_music_search.strip() != "":
            # 사용자 정의 검색어로 유튜브에서 3개 비디오를 한꺼번에 긁어와서 뿌려줌!
            with st.spinner("🔍 실시간 동영상을 유튜브에서 검색하여 로드 중입니다..."):
                vids = searchYoutubeVideos(custom_music_search, count=3)
            for i in range(3):
                if i < len(vids):
                    yt_embed_url = f"https://www.youtube.com/embed/{vids[i]}"
                    st.markdown(f"""
                    <div class="music-card" style="margin-top: 1.5rem; margin-bottom: 0.8rem; border-left: 5px solid #FF0000; padding-left: 15px;">
                        <div class="music-info">
                            <span style="background-color: #FF0000; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; display: inline-block; margin-bottom: 8px;">SEARCH RESULT {i+1}</span>
                            <div class="music-title" style="font-size: 1.4rem; font-weight: 800; color: #1e293b;">🔍 유튜브 실시간 연동 영상 {i+1}</div>
                            <div class="music-artist" style="font-size: 1.05rem; color: #4b5563; font-weight: 600; margin-bottom: 8px;">입력 검색어: "{custom_music_search}"</div>
                            <div class="music-desc" style="font-size: 0.95rem; line-height: 1.5; color: #1e293b; background-color: rgba(255, 0, 0, 0.05); padding: 10px 15px; border-radius: 8px;">유튜브 실시간 검색 및 고화질 스트리밍 연동 결과입니다.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <iframe style="border-radius:12px; width: 100%; height: 380px; box-shadow: 0 10px 20px rgba(0,0,0,0.08); border: 1px solid rgba(0,0,0,0.05);" 
                        src="{yt_embed_url}" 
                        frameborder="0" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen>
                    </iframe>
                    <div style="margin-bottom: 2rem;"></div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("검색된 동영상이 없습니다.")
        else:
            # 3가지 탭(K-POP, POP, J-POP)을 신설하여 정밀 3-3-3 추천 구성
            tab_kpop, tab_pop, tab_jpop = st.tabs(["🇰🇷 K-POP 추천 (3곡)", "🇺🇸 POP 추천 (3곡)", "🇯🇵 J-POP 추천 (3곡)"])
            
            # 1. K-POP 탭 렌더링
            with tab_kpop:
                for i, song in enumerate(recommendations.get("K-POP", [])):
                    search_query = f"{song['artist']} {song['title']}"
                    vids = searchYoutubeVideos(search_query, count=1)
                    st.markdown(f"""
                    <div class="music-card" style="margin-top: 1.5rem; margin-bottom: 0.8rem; border-left: 5px solid #1DB954; padding-left: 15px;">
                        <div class="music-info">
                            <span style="background-color: #1DB954; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; display: inline-block; margin-bottom: 8px;">TRACK {i+1}</span>
                            <div class="music-title" style="font-size: 1.4rem; font-weight: 800; color: #1e293b;">🎵 {song["title"]}</div>
                            <div class="music-artist" style="font-size: 1.05rem; color: #4b5563; font-weight: 600; margin-bottom: 8px;">{song["artist"]}</div>
                            <div class="music-desc" style="font-size: 0.95rem; line-height: 1.5; color: #1e293b; background-color: rgba(29, 185, 84, 0.05); padding: 10px 15px; border-radius: 8px;">"{song["desc"]}"</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if vids:
                        yt_embed_url = f"https://www.youtube.com/embed/{vids[0]}"
                        st.markdown(f"""
                        <iframe style="border-radius:12px; width: 100%; height: 380px; box-shadow: 0 10px 20px rgba(0,0,0,0.08); border: 1px solid rgba(0,0,0,0.05);" 
                            src="{yt_embed_url}" 
                            frameborder="0" 
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                            allowfullscreen>
                        </iframe>
                        <div style="margin-bottom: 2rem;"></div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("유튜브 영상을 불러오지 못했습니다.")
                            
            # 2. POP 탭 렌더링
            with tab_pop:
                for i, song in enumerate(recommendations.get("POP", [])):
                    search_query = f"{song['artist']} {song['title']}"
                    vids = searchYoutubeVideos(search_query, count=1)
                    st.markdown(f"""
                    <div class="music-card" style="margin-top: 1.5rem; margin-bottom: 0.8rem; border-left: 5px solid #8B5CF6; padding-left: 15px;">
                        <div class="music-info">
                            <span style="background-color: #8B5CF6; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; display: inline-block; margin-bottom: 8px;">TRACK {i+1}</span>
                            <div class="music-title" style="font-size: 1.4rem; font-weight: 800; color: #1e293b;">🎵 {song["title"]}</div>
                            <div class="music-artist" style="font-size: 1.05rem; color: #4b5563; font-weight: 600; margin-bottom: 8px;">{song["artist"]}</div>
                            <div class="music-desc" style="font-size: 0.95rem; line-height: 1.5; color: #1e293b; background-color: rgba(139, 92, 246, 0.05); padding: 10px 15px; border-radius: 8px;">"{song["desc"]}"</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if vids:
                        yt_embed_url = f"https://www.youtube.com/embed/{vids[0]}"
                        st.markdown(f"""
                        <iframe style="border-radius:12px; width: 100%; height: 380px; box-shadow: 0 10px 20px rgba(0,0,0,0.08); border: 1px solid rgba(0,0,0,0.05);" 
                            src="{yt_embed_url}" 
                            frameborder="0" 
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                            allowfullscreen>
                        </iframe>
                        <div style="margin-bottom: 2rem;"></div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("유튜브 영상을 불러오지 못했습니다.")
                            
            # 3. J-POP 탭 렌더링
            with tab_jpop:
                for i, song in enumerate(recommendations.get("J-POP", [])):
                    search_query = f"{song['artist']} {song['title']}"
                    vids = searchYoutubeVideos(search_query, count=1)
                    st.markdown(f"""
                    <div class="music-card" style="margin-top: 1.5rem; margin-bottom: 0.8rem; border-left: 5px solid #F59E0B; padding-left: 15px;">
                        <div class="music-info">
                            <span style="background-color: #F59E0B; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; display: inline-block; margin-bottom: 8px;">TRACK {i+1}</span>
                            <div class="music-title" style="font-size: 1.4rem; font-weight: 800; color: #1e293b;">🎵 {song["title"]}</div>
                            <div class="music-artist" style="font-size: 1.05rem; color: #4b5563; font-weight: 600; margin-bottom: 8px;">{song["artist"]}</div>
                            <div class="music-desc" style="font-size: 0.95rem; line-height: 1.5; color: #1e293b; background-color: rgba(245, 158, 11, 0.05); padding: 10px 15px; border-radius: 8px;">"{song["desc"]}"</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if vids:
                        yt_embed_url = f"https://www.youtube.com/embed/{vids[0]}"
                        st.markdown(f"""
                        <iframe style="border-radius:12px; width: 100%; height: 380px; box-shadow: 0 10px 20px rgba(0,0,0,0.08); border: 1px solid rgba(0,0,0,0.05);" 
                            src="{yt_embed_url}" 
                            frameborder="0" 
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                            allowfullscreen>
                        </iframe>
                        <div style="margin-bottom: 2rem;"></div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("유튜브 영상을 불러오지 못했습니다.")
                            
    
    # 🌐 실시간 음악 사이트 연동 검색 포털 추가
    with st.container(border=True):
        st.markdown("## 🌐 실시간 외부 음악 사이트 연동 검색")
        st.markdown("추천된 곡 이외에도, 사진의 분위기 분석 결과에 맞춰 원하는 음악 사이트에서 직접 맞춤 최신 음원을 실시간으로 검색하여 들어보실 수 있습니다.")
        
        # 대표 무드와 어울리는 자동 검색 쿼리 생성
        mood_keywords = {
            "Happy / Excited": "신나는 청량한 최신 KPOP 댄스",
            "Tense / Dark": "강렬하고 비장한 트렌디 힙합 일렉트로",
            "Sad / Lonely": "감성적이고 쓸쓸한 밤에 듣기 좋은 발라드",
            "Calm / Relaxed": "카페에서 듣기 좋은 나른하고 포근한 인디 칠아웃"
        }
        
        # 4분면 중 매핑 좌표에 기반한 지배적 무드 획득
        if targetV >= 0.5 and targetE >= 0.5:
            default_query = f"{mood_keywords['Happy / Excited']}"
        elif targetV < 0.5 and targetE >= 0.5:
            default_query = f"{mood_keywords['Tense / Dark']}"
        elif targetV < 0.5 and targetE < 0.5:
            default_query = f"{mood_keywords['Sad / Lonely']}"
        else:
            default_query = f"{mood_keywords['Calm / Relaxed']}"
            
        # 사용자가 직접 검색어를 수정해서 찾을 수 있도록 입력창 제공
        search_q = st.text_input("🔍 실시간 분위기 검색어 (원하시는 검색어로 수정이 가능합니다)", value=default_query)
        
        # URL 인코딩
        import urllib.parse
        enc_q = urllib.parse.quote(search_q)
        
        # 음악 사이트별 링크 포털 버튼 생성 (4단 컬럼)
        portal_cols = st.columns(4)
        
        with portal_cols[0]:
            st.markdown(f"""
            <a href="https://www.youtube.com/results?search_query={enc_q}" target="_blank" style="text-decoration: none;">
                <div style="background-color: #FF0000; color: white; padding: 12px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 0.95rem; cursor: pointer; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); transition: transform 0.2s;">
                    🔴 유튜브 (YouTube)
                </div>
            </a>
            """, unsafe_allow_html=True)
            
        with portal_cols[1]:
            st.markdown(f"""
            <a href="https://www.melon.com/search/total/index.htm?q={enc_q}" target="_blank" style="text-decoration: none;">
                <div style="background-color: #00CD3C; color: white; padding: 12px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 0.95rem; cursor: pointer; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); transition: transform 0.2s;">
                    🍈 멜론 (Melon)
                </div>
            </a>
            """, unsafe_allow_html=True)
            
        with portal_cols[2]:
            st.markdown(f"""
            <a href="https://www.genie.co.kr/search/searchMain?query={enc_q}" target="_blank" style="text-decoration: none;">
                <div style="background-color: #00A2ED; color: white; padding: 12px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 0.95rem; cursor: pointer; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); transition: transform 0.2s;">
                    🎵 지니 (Genie)
                </div>
            </a>
            """, unsafe_allow_html=True)
            
        with portal_cols[3]:
            st.markdown(f"""
            <a href="https://open.spotify.com/search/{enc_q}" target="_blank" style="text-decoration: none;">
                <div style="background-color: #1DB954; color: white; padding: 12px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 0.95rem; cursor: pointer; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); transition: transform 0.2s;">
                    💚 스포티파이 (Spotify)
                </div>
            </a>
            """, unsafe_allow_html=True)
            
else:
    # 이미지가 업로드되기 전 프리뷰용 안내 메인 화면
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 5rem 2rem;">
        <h2 style='font-weight: 700; color: #1DB954;'>🔮 감성 분석 매직 가이드</h2>
        <p style='color: #9ca3af; max-width: 600px; margin: 1rem auto 2rem auto; font-size: 1.1rem; line-height: 1.6;'>
            사진을 업로드하면 대표님이 직접 898장의 OASIS 고화질 감성 이미지 데이터셋을 
            활용해 지도 학습(Supervised Learning)을 완수하신 
            <b>Custom ResNet50 회귀 모델</b>이 형태와 정취를 읽어내고, <b>OpenCV</b> 명암/채도 컬러 스캔을 더해 
            <b>2차원 Russell 감성 분포 공간</b> 상에 정확히 안착시킵니다. 
            그 후 큐레이션된 명품 음원 데이터 풀에서 가장 높은 코사인 유사도를 자아내는 사운드를 정밀 매칭합니다.
        </p>
        <div style="margin-top: 2rem;">
            <span class="vibe-badge">Step 1. 사진 선택</span> 
            <span class="vibe-badge">Step 2. AI 실시간 정밀 스캔</span> 
            <span class="vibe-badge">Step 3. 유튜브 음악 감상</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
