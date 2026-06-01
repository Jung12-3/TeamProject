import cv2          # 웹캠을 제어하고 화면에 글씨나 도형을 그리기 위한 라이브러리
import mediapipe as mp # 사람의 손 관절을 인식해주는 구글의 인공지능 라이브러리
import time         # 카운트다운 타이머를 만들기 위한 시간 계산 라이브러리
import math         # 손가락 마디 사이의 거리를 계산하기 위한 수학 라이브러리


# [이벤트] 마우스 클릭 감지 함수 (버튼을 누를 수 있게 해줌)
exit_flag = False      # 프로그램을 끌지 말지 결정하는 스위치
game_mode = "WINNER"   # 게임의 기본 목표 (기본값: 이긴 사람 찾기)

def mouse_click(event, x, y, flags, param):
    global exit_flag, game_mode
    # 마우스 왼쪽 버튼이 '클릭' 되었을 때만 작동
    if event == cv2.EVENT_LBUTTONDOWN:
        
        # 1. 우측 상단의 [EXIT] 버튼 영역(좌표)을 클릭하면 종료
        if 1120 <= x <= 1260 and 15 <= y <= 65:
            exit_flag = True # 종료 스위치를 켬 (반복문이 멈춤)
            
        # 2. 좌측 상단의 [MODE] 버튼 영역(좌표)을 클릭하면 승리 or 패배자 모드 변경
        elif 20 <= x <= 320 and 75 <= y <= 125:
            # 현재 모드가 WINNER면 LOSER로, LOSER면 WINNER로 바꿈
            if game_mode == "WINNER": game_mode = "LOSER"
            else: game_mode = "WINNER"


# 1. MediaPipe 손 인식 AI 모델 초기화
mp_hands = mp.solutions.hands       # 21개의 손 관절 좌표를 찾는 AI 모델
mp_drawing = mp.solutions.drawing_utils # 찾은 관절을 화면에 예쁘게 그려주는 도구

hands = mp_hands.Hands(
    max_num_hands=6,               # 최대 6명의 손을 동시에 인식
    min_detection_confidence=0.7,  # 손을 처음 찾을 때 70% 이상 확신해야 손으로 인정
    min_tracking_confidence=0.7    # 찾은 손을 따라다닐 때의 민감도 설정
)


# 2. 게임 상태 및 전적을 기록할 변수들 준비
game_state = 0   # 게임의 현재 상태 (0:대기, 1:3초 카운트다운, 2:손 내밀기 대기, 3:결과 창)
start_time = 0   # 카운트다운용 시간을 담아둘 변수
shoot_time = 0   # 누군가 처음 손을 낸 순간부터 '0.5초 유예시간'을 재기 위한 타이머

total_games = 0  # 총 진행한 게임 수
# 플레이어 1번~6번까지의 승리(Wins)와 패배(Losses) 횟수를 담아둘 딕셔너리(사전)
player_wins = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}   
player_losses = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0} 

match_result_1 = "" # 메인 화면에 띄울 글씨 (예: "WINNERS: P1, P2")
match_result_2 = "" # 서브 화면에 띄울 글씨 (예: "안 내서 진 사람: P3")
targets_list = []   # 모드에 따라 목표가 된 사람(승자 혹은 패자)의 번호 명단
invalid_pids = []   # 안 내거나 늦게 내서 패널티를 받은 사람의 번호 명단

rps_list = ["ROCK", "PAPER", "SCISSORS"] # 정상적인 가위바위보 목록


# 3. 제스처 판별 로직 (권총 가위 + 360도 방향 인식)
# 피타고라스 정리를 이용해 두 점(p1, p2) 사이의 거리를 계산하는 함수
def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

# 21개의 손 관절 데이터를 분석해서 무슨 모양인지 알아내는 함수
def get_gesture(hand_landmarks):
    open_fingers = [] # 손가락이 펴졌는지(1), 접혔는지(0) 담아둘 리스트
    
    # 5개 손가락의 [끝 마디]와 [안쪽 두 번째 마디] 번호
    finger_tips = [4, 8, 12, 16, 20] # 엄지, 검지, 중지, 약지, 새끼 끝
    finger_pips = [2, 6, 10, 14, 18] # 엄지, 검지, 중지, 약지, 새끼 안쪽
    
    # 손가락 거리 측정의 '기준점'이 되는 손목(0번)의 위치
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # 5개 손가락을 하나씩 확인
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[pip_idx]
        
        # 끝 마디가 안쪽 마디보다 손목에서 더 멀리 떨어져 있다면 -> 손가락을 폈다!
        if get_dist(wrist, tip) > get_dist(wrist, pip): open_fingers.append(1) 
        else: open_fingers.append(0) 
            
    # [가위 1: V자형] 엄지 빼고(open_fingers[1:]), 검지와 중지만 펴진 상태
    is_standard_scissors = (open_fingers[1:] == [1, 1, 0, 0])
    # [가위 2: 권총형] 엄지와 검지만 펴진 상태
    is_gun_scissors = (open_fingers == [1, 1, 0, 0, 0])
    
    if is_standard_scissors or is_gun_scissors: return "SCISSORS" # 둘 중 하나면 가위
    elif open_fingers[1:] == [0, 0, 0, 0]: return "ROCK"          # 4개가 다 접히면 바위
    elif open_fingers.count(1) >= 4: return "PAPER"               # 4개 이상 펴지면 보
    else: return "UNKNOWN"                                        # 나머지는 이상한 모양 (나중에 시간 지나면 패배처리)


# 4. [핵심] 다인용 판정 로직 (모드 적용 + 안 낸 사람 색출)
def get_multiplayer_result(player_choices_dict, current_mode):
    # 1. 가위바위보를 제대로 낸 사람(valid)과, 안 내거나 이상하게 낸 사람(invalid)을 분류함
    valid_choices = {pid: g for pid, g in player_choices_dict.items() if g in rps_list}
    invalid_pids = [pid for pid, g in player_choices_dict.items() if g not in rps_list]
    
    # 안 낸 사람이 있다면 화면에 띄울 '패널티 경고 문구'를 미리 만들어둠
    penalty_text = ""
    if invalid_pids:
        penalty_text = f"[PENALTY] Late: P{', P'.join(map(str, invalid_pids))} LOSE!"
        
    # 만약 아무도 안 냈다면 무승부 처리
    if not valid_choices:
        return "DRAW! (NO ONE PLAYED)", penalty_text, [], invalid_pids

    # 사람들이 낸 손 모양의 '종류'만 중복을 없애고 모아봄 (예: 바위와 가위만 나왔다면 2종류)
    unique_gestures = set(valid_choices.values())
    
    # 2. [무승부 조건] 전부 같은 걸 냈거나(1종류), 가위/바위/보가 다 나왔거나(3종류)
    if len(unique_gestures) == 1 or len(unique_gestures) == 3:
        return f"DRAW! (IN {current_mode} MODE)", penalty_text, [], invalid_pids
            
    # 3. 딱 2종류만 나왔을 때 승패 결정
    winning_gesture, losing_gesture = "", ""
    if unique_gestures == {"ROCK", "SCISSORS"}:
        winning_gesture, losing_gesture = "ROCK", "SCISSORS"
    elif unique_gestures == {"SCISSORS", "PAPER"}:
        winning_gesture, losing_gesture = "SCISSORS", "PAPER"
    elif unique_gestures == {"PAPER", "ROCK"}:
        winning_gesture, losing_gesture = "PAPER", "ROCK"
    
    # 4. 사용자가 선택한 모드(승자 찾기 vs 패자 찾기)에 따라 정답 대상(Targets)을 다르게 뽑음
    if current_mode == "WINNER": # WINNER 모드일 경우
        # 이긴 모양을 낸 사람들의 번호만 추려냄
        targets = [pid for pid, g in valid_choices.items() if g == winning_gesture]
        target_text = "WINNERS: P" + ", P".join(map(str, targets))
    else: # LOSER 모드일 경우
        # 진 모양을 낸 사람들의 번호만 추려냄
        targets = [pid for pid, g in valid_choices.items() if g == losing_gesture]
        target_text = "LOSERS: P" + ", P".join(map(str, targets))
        
    return target_text, penalty_text, targets, invalid_pids


# 5. 메인 루프 (카메라를 켜고 영상을 실시간으로 처리함)
cap = cv2.VideoCapture(0) # 0번 카메라(기본 웹캠) 연결
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 화면 가로 크기 HD
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 화면 세로 크기 HD

window_name = '가위바위보 게임'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_click) # 마우스 클릭 기능 연결

# 카메라가 켜져 있고 종료 스위치가 안 눌린 동안 무한 반복
while cap.isOpened() and not exit_flag:
    ret, frame = cap.read() # 카메라에서 사진 한 장 찍기
    if not ret: break
    
    frame = cv2.flip(frame, 1) # 거울처럼 좌우 반전
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB 색상으로 변환
    
    results = hands.process(rgb_frame) # AI 모델에게 손을 찾아달라고 명령
    
    player_gestures = {} # 이번 화면에 있는 플레이어들의 번호와 모양을 담을 사전
    num_detected = 0     # 화면에 인식된 총 사람 수
    
    # 화면에 손이 1개 이상 인식되었다면
    if results.multi_hand_landmarks:
        hands_data = []
        for hand_landmarks in results.multi_hand_landmarks:
            # 뼈대와 관절을 화면에 그려줌
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 왼쪽부터 순서를 매기기 위해 '손목의 X(가로) 좌표'를 같이 저장
            wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            hands_data.append((wrist_x, hand_landmarks))
            
        # X 좌표가 작은 순(화면 왼쪽부터 1, 오른쪽으로) 정렬
        hands_data.sort(key=lambda x: x[0])
        num_detected = len(hands_data)
        
        # 정렬된 순서대로 P1, P2... 번호를 부여하고 글씨를 화면에 띄움
        for i, (wrist_x, hand_landmarks) in enumerate(hands_data):
            player_id = i + 1 
            gesture = get_gesture(hand_landmarks) # 손 모양 판별
            player_gestures[player_id] = gesture  # 사전(Dictionary)에 저장
            
            # 글씨를 띄울 위치를 계산 (손목 좌표에서 50픽셀 위로 올려서 그림)
            text_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * 1280) - 50
            text_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * 720) - 50
            
            # 이상하게 내거나 안 낸 사람(UNKNOWN)은 빨간색으로, 정상은 노란색으로 글씨를 씀
            color = (0, 0, 255) if gesture == "UNKNOWN" else (255, 200, 0)
            cv2.putText(frame, f"P{player_id}: {gesture}", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    
    # 6. 최상단 통계 바 출력
    # 현재 게임 모드에 따라서 보여주는 점수를 '승리 횟수' 혹은 '패배 횟수'로 바꿈
    if game_mode == "WINNER":
        stats_text = f"Total: {total_games} | WINS -> P1:{player_wins[1]} P2:{player_wins[2]} P3:{player_wins[3]} P4:{player_wins[4]} P5:{player_wins[5]} P6:{player_wins[6]}"
    else:
        stats_text = f"Total: {total_games} | LOSSES -> P1:{player_losses[1]} P2:{player_losses[2]} P3:{player_losses[3]} P4:{player_losses[4]} P5:{player_losses[5]} P6:{player_losses[6]}"
        
    # 글씨가 잘 보이게 검은색 배경 박스를 그리고 텍스트를 올림
    cv2.rectangle(frame, (0, 0), (1280, 60), (0, 0, 0), -1)
    cv2.putText(frame, stats_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    
    # 7. 게임 진행 상태 및 패널티 흐름 제어
    if game_state == 0:
        # [상태 0] 대기: 스페이스바를 누르라고 안내함
        cv2.putText(frame, "Press 'SPACE' to Start Match", (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        shoot_time = 0 # 0.5초 유예시간 타이머 초기화
        
    elif game_state == 1:
        # [상태 1] 카운트다운(3초)
        elapsed = time.time() - start_time
        countdown = 3 - int(elapsed)
        if countdown > 0:
            cv2.putText(frame, str(countdown), (600, 400), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), 15)
        else:
            game_state = 2 # 3초 지나면 가위바위보 결과를 보여줌
            
    elif game_state == 2:
        # [상태 2] SHOOT: 모두가 손을 내밀기를 기다림
        cv2.putText(frame, "SHOOT!", (500, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 8)
        
        # 현재 화면에서 정상적인 가위/바위/보를 낸 사람의 숫자를 셈
        valid_count = sum(1 for g in player_gestures.values() if g in rps_list)
        
        # 안 내면 진다!! (안 내면 패배처리)
        if num_detected >= 2 and valid_count > 0:
            # 제일 먼저 낸 사람이 확인된 순간, 타이머(0.5초)를 재기 시작함
            if shoot_time == 0:
                shoot_time = time.time()
                
            # 남은 유예 시간을 계산해서 화면에 빨간색으로 띄움
            time_left = 0.5 - (time.time() - shoot_time)
            if time_left > 0:
                cv2.putText(frame, f"LOCKING IN: {time_left:.1f}s", (480, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # 0.5초가 지나면 -> 즉시 사진을 찍고 승패를 판정
            if time.time() - shoot_time > 0.5:
                # 판정 함수 호출
                match_result_1, match_result_2, targets_list, invalid_pids = get_multiplayer_result(player_gestures, game_mode)
                
                total_games += 1 # 게임 횟수 증가
                
                # 모드에 따라 통계 점수를 올려줌
                if game_mode == "WINNER":
                    for w in targets_list: player_wins[w] += 1
                else:
                    for l in targets_list: player_losses[l] += 1
                    
                # 늦게 내거나 이상하게 낸 사람은 모드에 상관없이 무조건 패배 횟수 증가
                for p in invalid_pids:
                    player_losses[p] += 1
                
                game_state = 3 # 결과를 보여주는 상태로 변경
                start_time = time.time() # 결과창 유지 시간 타이머 시작

    elif game_state == 3:
        # [상태 3] 결과 표시 (4초 동안 화면 중앙에 띄워줌)
        if time.time() - start_time < 4.0:
            # 모드와 결과에 따라 글씨 색상을 다르게 칠함
            if "DRAW" in match_result_1: color = (0, 255, 255)      # 무승부 = 노란색
            elif game_mode == "WINNER": color = (0, 255, 0)         # 승자 모드 = 초록색
            else: color = (0, 0, 255)                               # 패자 모드 = 빨간색
            
            # 메인 결과 출력
            cv2.putText(frame, match_result_1, (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 6)
            
            # 만약 안 내서 패널티 받은 사람이 있다면 빨간색으로 그 밑에 경고문 출력
            if match_result_2:
                cv2.putText(frame, match_result_2, (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
        else:
            game_state = 0 # 4초 지나면 다시 처음 대기 상태


    # 8. 마우스로 누를 수 있는 UI 버튼 그리기 (MODE / EXIT)
    # [좌측 상단] MODE 전환 버튼 (WINNER면 초록색, LOSER면 빨간색으로 칠함)
    mode_color = (0, 200, 0) if game_mode == "WINNER" else (0, 0, 200)
    cv2.rectangle(frame, (20, 75), (320, 125), mode_color, -1)
    cv2.putText(frame, f"MODE: {game_mode}", (35, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # [우측 상단] EXIT 버튼 (빨간 박스에 흰 글씨)
    cv2.rectangle(frame, (1120, 15), (1260, 65), (0, 0, 255), -1) 
    cv2.putText(frame, "EXIT", (1155, 48), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            
    # 완성된 그림을 최종적으로 화면에 띄움
    cv2.imshow(window_name, frame)

    # 키보드 입력 감지 (1밀리초 대기)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # 스페이스바 누르면 카운트다운 시작
        if game_state == 0:
            game_state = 1
            start_time = time.time()
    elif key == ord('q'): # 'q'를 누르면 종료
        break

# 프로그램 종료 시 카메라 연결을 끊고 창을 닫음
cap.release()
cv2.destroyAllWindows()