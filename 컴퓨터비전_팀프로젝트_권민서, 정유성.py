import cv2          # 웹캠을 제어하고 화면에 글씨나 도형을 그리기 위한 라이브러리
import mediapipe as mp # 사람의 손 관절을 인식해주는 구글의 인공지능 라이브러리
import time         # 카운트다운 타이머를 만들기 위한 시간 계산 라이브러리
import math         # 손가락 마디 사이의 거리를 계산하기 위한 수학 라이브러리

# ---------------------------------------------------------
# [이벤트] 마우스 클릭 감지 함수 (버튼을 누를 수 있게 해줌)
# ---------------------------------------------------------
exit_flag = False      # 프로그램을 끌지 말지 결정하는 스위치
game_mode = "WINNER"   # 게임의 기본 목표 (기본값: 이긴 사람 찾기)

def mouse_click(event, x, y, flags, param):
    global exit_flag, game_mode
    # 마우스 왼쪽 버튼이 '클릭' 되었을 때만 작동
    if event == cv2.EVENT_LBUTTONDOWN:
        
        # 1. 우측 상단의 [EXIT] 버튼 영역(좌표)을 클릭했다면?
        if 1120 <= x <= 1260 and 15 <= y <= 65:
            exit_flag = True # 종료 스위치를 켬 (반복문이 멈춤)
            
        # 2. 좌측 상단의 [MODE] 버튼 영역(좌표)을 클릭했다면?
        elif 20 <= x <= 320 and 75 <= y <= 125:
            # 현재 모드가 WINNER면 LOSER로, LOSER면 WINNER로 바꿈
            if game_mode == "WINNER": game_mode = "LOSER"
            else: game_mode = "WINNER"


# 1. MediaPipe 손 인식 AI 모델 최적화 (근거리 다인용 셋팅)
mp_hands = mp.solutions.hands       # 21개의 손 관절 좌표를 찾는 AI 모델
mp_drawing = mp.solutions.drawing_utils # 찾은 관절을 화면에 예쁘게 그려주는 도구

hands = mp_hands.Hands(
    model_complexity=1,            # 0(빠름)과 1(정확함) 중 정확도 위주인 1로 설정
    max_num_hands=6,               # 최대 6명의 손을 동시에 인식함
    # 근거리에서 손이 겹칠 때를 대비해 민감도(Confidence) 컷트라인을 0.5로 설정
    min_detection_confidence=0.5,  
    min_tracking_confidence=0.5    
)


# 2. 게임 상태 및 전적을 기록할 변수들 준비
game_state = 0   # 게임 상태 (0:대기, 1:카운트다운, 2:손 내밀기 대기, 3:결과 창)
start_time = 0   # 카운트다운용 시간을 담아둘 변수
shoot_time = 0   # 누군가 처음 손을 낸 순간부터 '0.5초 유예시간'을 재기 위한 타이머

total_games = 0  # 총 진행한 게임 수
# 플레이어 1번~6번까지의 승리와 패배 횟수를 담아둘 사전(Dictionary)
player_wins = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}   
player_losses = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0} 

match_result_1 = "" # 메인 화면에 띄울 글씨 (예: "WINNERS: P1, P2")
match_result_2 = "" # 서브 화면에 띄울 글씨 (예: "안 내서 진 사람: P3")
targets_list = []   # 모드에 따라 목표가 된 사람(승자 혹은 패자)의 번호 명단
invalid_pids = []   # 안 내거나 늦게 내서 패널티를 받은 사람의 번호 명단

rps_list = ["ROCK", "PAPER", "SCISSORS"] # 정상적인 가위바위보 목록


# 3. 인공지능 제스처 판별 로직 (권총 가위 + 360도 방향 인식)
def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y) # 피타고라스 정리로 두 점 사이 거리 계산

def get_gesture(hand_landmarks):
    open_fingers = [] # 손가락이 펴졌는지(1), 접혔는지(0) 담아둘 리스트
    
    # 5개 손가락의 [끝 마디]와 [안쪽 두 번째 마디] 번호
    finger_tips = [4, 8, 12, 16, 20] 
    finger_pips = [2, 6, 10, 14, 18] 
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST] # 거리 측정 기준점(손목)
    
    # 5개 손가락을 하나씩 확인하여 펴짐/접힘 여부를 리스트에 넣음
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[pip_idx]
        if get_dist(wrist, tip) > get_dist(wrist, pip): open_fingers.append(1) 
        else: open_fingers.append(0) 
            
    # [가위 1: V자형] 엄지 상태 무시([1:] 사용), 검지와 중지만 펴진 상태
    is_standard_scissors = (open_fingers[1:] == [1, 1, 0, 0])
    # [가위 2: 권총형] 엄지와 검지만 펴진 상태
    is_gun_scissors = (open_fingers == [1, 1, 0, 0, 0])
    
    if is_standard_scissors or is_gun_scissors: return "SCISSORS" # 둘 중 하나면 가위
    elif open_fingers[1:] == [0, 0, 0, 0]: return "ROCK"          # 엄지 제외 4개가 다 접히면 바위
    elif open_fingers.count(1) >= 4: return "PAPER"               # 4개 이상 펴지면 보
    else: return "UNKNOWN"                                        # 나머지는 이상한 모양


# 4. [핵심] 다인용 판정 로직 (모드 적용 + 안 낸 사람 색출)
def get_multiplayer_result(player_choices_dict, current_mode):
    # 1. 제대로 낸 사람(valid)과 안 내거나 이상하게 낸 사람(invalid)을 분류
    valid_choices = {pid: g for pid, g in player_choices_dict.items() if g in rps_list}
    invalid_pids = [pid for pid, g in player_choices_dict.items() if g not in rps_list]
    
    # 안 낸 사람이 있다면 띄울 '패널티 경고 문구' 생성
    penalty_text = ""
    if invalid_pids:
        penalty_text = f"[PENALTY] Late: P{', P'.join(map(str, invalid_pids))} LOSE!"
        
    if not valid_choices: # 아무도 안 냈다면 무승부 반환
        return "DRAW! (NO ONE PLAYED)", penalty_text, [], invalid_pids

    # 낸 손 모양들의 '종류'만 중복 없이 모음 (예: 바위와 가위만 나왔다면 길이는 2)
    unique_gestures = set(valid_choices.values())
    
    # 2. [무승부 조건] 전부 같은 걸 냈거나(1종), 가위/바위/보가 다 나왔거나(3종)
    if len(unique_gestures) == 1 or len(unique_gestures) == 3:
        return f"DRAW! (IN {current_mode} MODE)", penalty_text, [], invalid_pids
            
    # 3. 딱 2종류만 나왔을 때 승패 결정
    winning_gesture, losing_gesture = "", ""
    if unique_gestures == {"ROCK", "SCISSORS"}: winning_gesture, losing_gesture = "ROCK", "SCISSORS"
    elif unique_gestures == {"SCISSORS", "PAPER"}: winning_gesture, losing_gesture = "SCISSORS", "PAPER"
    elif unique_gestures == {"PAPER", "ROCK"}: winning_gesture, losing_gesture = "PAPER", "ROCK"
    
    # 4. 모드(WINNER vs LOSER)에 따라 정답 대상을 다르게 뽑음
    if current_mode == "WINNER":
        targets = [pid for pid, g in valid_choices.items() if g == winning_gesture]
        target_text = "WINNERS: P" + ", P".join(map(str, targets))
    else: 
        targets = [pid for pid, g in valid_choices.items() if g == losing_gesture]
        target_text = "LOSERS: P" + ", P".join(map(str, targets))
        
    return target_text, penalty_text, targets, invalid_pids


# 5. 메인 루프 (카메라 영상 실시간 처리)
cap = cv2.VideoCapture(0) # 0번 카메라 연결
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 화면 크기 HD
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  

window_name = 'Ultimate Multiplayer RPS Referee'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_click) # 마우스 클릭 이벤트 연결

while cap.isOpened() and not exit_flag:
    ret, frame = cap.read() # 사진 한 장 찍기
    if not ret: break
    
    frame = cv2.flip(frame, 1) # 거울처럼 좌우 반전
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # AI용 색상 변환
    
    results = hands.process(rgb_frame) # AI 모델에게 손 찾기 명령
    
    player_gestures = {} # 화면 속 플레이어 번호와 모양을 담을 임시 사전
    num_detected = 0     # 현재 화면에 인식된 총 사람 수
    
    if results.multi_hand_landmarks:
        hands_data = []
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 왼쪽부터 순서를 매기기 위해 '손목의 X(가로) 좌표'를 저장
            wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            hands_data.append((wrist_x, hand_landmarks))
            
        hands_data.sort(key=lambda x: x[0]) # X 좌표 작은 순(화면 왼쪽부터) 정렬
        num_detected = len(hands_data)
        
        for i, (wrist_x, hand_landmarks) in enumerate(hands_data):
            player_id = i + 1 
            gesture = get_gesture(hand_landmarks) 
            player_gestures[player_id] = gesture  
            
            # 글씨 띄울 위치 계산 (손목 좌표에서 50픽셀 위)
            text_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * 1280) - 50
            text_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * 720) - 50
            
            # 안 낸 사람(UNKNOWN)은 빨간색, 정상은 노란색으로 글씨 출력
            color = (0, 0, 255) if gesture == "UNKNOWN" else (255, 200, 0)
            cv2.putText(frame, f"P{player_id}: {gesture}", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    
    # 6. 상단 통계 바 출력
    if game_mode == "WINNER":
        stats_text = f"Total: {total_games} | WINS -> P1:{player_wins[1]} P2:{player_wins[2]} P3:{player_wins[3]} P4:{player_wins[4]} P5:{player_wins[5]} P6:{player_wins[6]}"
    else:
        stats_text = f"Total: {total_games} | LOSSES -> P1:{player_losses[1]} P2:{player_losses[2]} P3:{player_losses[3]} P4:{player_losses[4]} P5:{player_losses[5]} P6:{player_losses[6]}"
        
    cv2.rectangle(frame, (0, 0), (1280, 60), (0, 0, 0), -1)
    cv2.putText(frame, stats_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    
    # 7. 게임 진행 상태(Game State) 및 타이머 제어
    if game_state == 0:
        cv2.putText(frame, "Press 'SPACE' to Start Match", (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        shoot_time = 0 # 유예시간 타이머 초기화
        
    elif game_state == 1: # 카운트다운 (3, 2, 1)
        elapsed = time.time() - start_time
        countdown = 3 - int(elapsed)
        if countdown > 0:
            cv2.putText(frame, str(countdown), (600, 400), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), 15)
        else:
            game_state = 2 
            
    elif game_state == 2: # SHOOT 대기
        cv2.putText(frame, "SHOOT!", (500, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 8)
        
        valid_count = sum(1 for g in player_gestures.values() if g in rps_list)
        
        # 화면에 2명 이상 있고, 최소 1명이라도 정상적으로 냈다면 타이머 시작
        if num_detected >= 2 and valid_count > 0:
            if shoot_time == 0:
                shoot_time = time.time()
                
            # 남은 시간(0.5초)을 화면에 붉은색으로 띄움
            time_left = 0.5 - (time.time() - shoot_time)
            if time_left > 0:
                cv2.putText(frame, f"LOCKING IN: {time_left:.1f}s", (480, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # 0.5초 경과 시 즉시 캡처 및 판정
            if time.time() - shoot_time > 0.5:
                match_result_1, match_result_2, targets_list, invalid_pids = get_multiplayer_result(player_gestures, game_mode)
                
                total_games += 1 
                
                # 모드에 따라 승리/패배 횟수 증가
                if game_mode == "WINNER":
                    for w in targets_list: player_wins[w] += 1
                else:
                    for l in targets_list: player_losses[l] += 1
                    
                # 패널티를 받은 사람은 무조건 '패배 횟수' 증가
                for p in invalid_pids:
                    player_losses[p] += 1
                
                game_state = 3 
                start_time = time.time() 

    elif game_state == 3: # 결과 화면 출력 (4초간)
        if time.time() - start_time < 4.0:
            if "DRAW" in match_result_1: color = (0, 255, 255)      # 무승부 = 노란색
            elif game_mode == "WINNER": color = (0, 255, 0)         # 승자 모드 = 초록색
            else: color = (0, 0, 255)                               # 패자 모드 = 빨간색
            
            cv2.putText(frame, match_result_1, (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 6)
            
            if match_result_2: # 패널티 안내문이 있으면 빨간색으로 함께 출력
                cv2.putText(frame, match_result_2, (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
        else:
            game_state = 0 

    
    # 8. 클릭 가능한 UI 버튼 그리기
    # [좌측 상단] MODE 전환 버튼
    mode_color = (0, 200, 0) if game_mode == "WINNER" else (0, 0, 200)
    cv2.rectangle(frame, (20, 75), (320, 125), mode_color, -1)
    cv2.putText(frame, f"MODE: {game_mode}", (35, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # [우측 상단] EXIT 버튼
    cv2.rectangle(frame, (1120, 15), (1260, 65), (0, 0, 255), -1) 
    cv2.putText(frame, "EXIT", (1155, 48), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            
    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # 스페이스바
        if game_state == 0:
            game_state = 1
            start_time = time.time()
    elif key == ord('q'): # q키 종료
        break

cap.release()
cv2.destroyAllWindows()