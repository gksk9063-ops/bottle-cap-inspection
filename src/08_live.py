"""v4: 폰(IP Webcam) 실시간 병뚜껑 검사 — cap_core.judge + 최근 N프레임 다수결.

키: 1 = 실제 '뚜껑 있음'으로 기록 / 2 = 실제 '뚜껑 없음'으로 기록
    s = 화면 저장 / q = 종료
기록은 reports/v4_live_log.csv 에 쌓이고, 종료할 때 정확도를 출력한다.
"""
import cv2, csv, os, time
from collections import Counter, deque
import cap_core as core

URL = "http://192.168.40.132:4747/video"   # ← 내 폰 IP Webcam 주소
VOTE_N = 15         # 최근 15프레임으로
VOTE_MIN = 10       # 같은 판정이 10번 이상이면 확정
COLORS = {"OK": (0, 200, 0), "NG": (0, 0, 255), "RETAKE": (0, 200, 255), "...": (200, 200, 200)}

os.makedirs("reports", exist_ok=True); os.makedirs("results/live", exist_ok=True)
log_path = "reports/v4_live_log.csv"
new_log = not os.path.exists(log_path)
log = open(log_path, "a", newline=""); w = csv.writer(log)
if new_log:
    w.writerow(["time", "gt", "final", "frame_verdict", "reason", "ratio", "tilt"])

cap = cv2.VideoCapture(URL)
if not cap.isOpened():
    raise SystemExit("연결 실패: 폰 주소와 같은 와이파이인지 확인하세요")

votes = deque(maxlen=VOTE_N)
n_save = 0
while True:
    ok, frame = cap.read()
    if not ok:
        print("프레임을 못 받음"); break
    img = core.resize_w1000(frame)
    r = core.judge(img)
    votes.append(r["verdict"])

    top, cnt = Counter(votes).most_common(1)[0]
    final = top if (len(votes) == VOTE_N and cnt >= VOTE_MIN) else "..."

    view = img.copy()
    if r["roi"]:
        x, y, bw, bh = r["roi"]
        cv2.rectangle(view, (x, y), (x + bw, y + bh), COLORS[r["verdict"]], 4)
    cv2.putText(view, final, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.5, COLORS[final], 8)
    info = f"frame {r['verdict']} {r['reason']}  ratio {r['ratio'] or 0:.2f}  tilt {r['tilt'] or 0:.1f}  vote {cnt}/{len(votes)}"
    cv2.putText(view, info, (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
    cv2.imshow("cap inspector v4", cv2.resize(view, None, fx=0.6, fy=0.6))

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("1"), ord("2")):
        gt = "OK" if key == ord("1") else "NG"
        w.writerow([time.strftime("%H:%M:%S"), gt, final, r["verdict"], r["reason"],
                    round(r["ratio"], 3) if r["ratio"] else "", round(r["tilt"], 1) if r["tilt"] is not None else ""])
        log.flush()
        print(f"기록: 정답 {gt} / 판정 {final}")
    elif key == ord("s"):
        n_save += 1
        cv2.imwrite(f"results/live/live_{n_save:02d}.png", view)
    elif key == ord("q"):
        break

cap.release(); cv2.destroyAllWindows(); log.close()

rows = list(csv.DictReader(open(log_path)))
judged = [r for r in rows if r["final"] in ("OK", "NG")]
correct = sum(r["final"] == r["gt"] for r in judged)
danger = sum(r["gt"] == "NG" and r["final"] == "OK" for r in judged)
print(f"[v4 실시간] 판정 {correct}/{len(judged)} · RETAKE/대기 {len(rows) - len(judged)} · 뚜껑 없는데 통과 {danger}건")
