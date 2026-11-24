import json
import os
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 檔案放在這支程式所在的資料夾 (不受執行時的工作目錄影響)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INFO_FILE = os.path.join(BASE_DIR, "info.json")
SAVE_FILE = os.path.join(BASE_DIR, "room_state.json")
EPS = 1e-9

# data 裡的資料一律維持 JSON 的格式 (list 和數字), 只有在做幾何運算時才暫時轉成 array。
# 這樣存檔可以直接 json.dump(data), 不需要 .tolist()。
#   房間: origin [x, y] / size [寬, 長, 高]
#   家具: pos [x, y] (家具中心, 房間內的相對座標) / size [寬, 長, 高] / yaw (度, 逆時針為正)
with open(INFO_FILE, "r", encoding="utf-8") as _f:
    data = json.load(_f)


def find_room(room_id):
    for room in data["rooms"]:
        if room["id"] == room_id:
            return room
    return None   # 找不到


def find_furniture(f_id, f_infos):
    for f in f_infos:
        if f["id"] == f_id:
            return f
    return None   # 找不到


def locate(f_id):
    """用家具 id 找出 (所在房間, 家具); 找不到就丟出 ValueError"""
    for room in data["rooms"]:
        f = find_furniture(f_id, room["furniture"])
        if f is not None:
            return room, f
    raise ValueError(f"Unknown furniture: {f_id}")


def room_bounds(room):
    """回傳房間邊界 (xmin, ymin, xmax, ymax)"""
    x, y = room["origin"]
    w, h = room["size"][0], room["size"][1]
    return x, y, x + w, y + h


def door_segment(room, door):
    """
    回傳門在牆上的兩個端點 ((xa, ya), (xb, yb)), 全域座標。
    offset 是門的起點離該面牆起點的距離 (南北牆從西往東量, 東西牆從南往北量)。
    """
    x0, y0, x1, y1 = room_bounds(room)
    o, w = door["offset"], door["width"]
    wall = door["wall"]
    if wall == "south":
        return (x0 + o, y0), (x0 + o + w, y0)
    if wall == "north":
        return (x0 + o, y1), (x0 + o + w, y1)
    if wall == "west":
        return (x0, y0 + o), (x0, y0 + o + w)
    if wall == "east":
        return (x1, y0 + o), (x1, y0 + o + w)
    raise ValueError(f"Unknown wall: {wall}")


def furniture_corners(room, f, pos=None, yaw=None):
    """
    由 pos / size / yaw 算出家具四個頂點 (全域座標, array, 形狀 (4, 2))。
    pos、yaw 可以另外指定, 用來計算「還沒套用的新位置」。
    """
    pos = f["pos"] if pos is None else pos
    yaw = f["yaw"] if yaw is None else yaw
    w, h = f["size"][0], f["size"][1]
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    rad = np.radians(yaw)
    rotation = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)],
    ])
    # list → array 在這裡轉換; 家具 pos 是房間內相對座標, 要加上房間 origin 才是全域座標
    center = np.array(pos, dtype=float) + np.array(room["origin"], dtype=float)
    return local @ rotation.T + center


def clamp_shift(corners, room):
    """
    計算把家具平移回房間內所需的位移 (dx, dy)。
    家具已在房間內則為 (0, 0); 超出的部分停留在邊界值。
    """
    rx0, ry0, rx1, ry1 = room_bounds(room)
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)

    dx = 0.0
    if x0 < rx0:
        dx = rx0 - x0
    elif x1 > rx1:
        dx = rx1 - x1

    dy = 0.0
    if y0 < ry0:
        dy = ry0 - y0
    elif y1 > ry1:
        dy = ry1 - y1
    return dx, dy


def polygons_overlap(a, b):
    """
    判斷兩個凸多邊形是否重疊 (分離軸定理, 旋轉後的家具也適用)。
    只是邊貼著邊、點貼著點不算重疊。
    """
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly                 # 每條邊的向量
        normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)  # 每條邊的法向量
        for n in normals:
            length = np.linalg.norm(n)
            if length == 0:
                continue
            n = n / length
            pa, pb = a @ n, b @ n                                # 兩個多邊形在此軸上的投影
            if pa.max() <= pb.min() + EPS or pb.max() <= pa.min() + EPS:
                return False                                     # 找到分離軸, 沒重疊
    return True


def find_overlap(room, f_id, corners):
    """檢查 corners (家具 f_id 的新位置) 是否與同房間其他家具重疊, 回傳重疊家具 id, 沒有則回傳 None"""
    for other in room["furniture"]:
        if other["id"] != f_id and polygons_overlap(corners, furniture_corners(room, other)):
            return other["id"]
    return None


def place(room, f, pos, yaw):
    """
    嘗試把家具放到 pos / yaw: 超出房間就推回邊界; 若會與其他家具重疊則不放。
    回傳擋住的家具 id, 成功更新則回傳 None。
    """
    pos = np.array(pos, dtype=float)
    corners = furniture_corners(room, f, pos, yaw)
    sx, sy = clamp_shift(corners, room)
    pos = pos + np.array([sx, sy])
    corners = corners + np.array([sx, sy])

    blocker = find_overlap(room, f["id"], corners)
    if blocker is None:
        # 寫回 data 時轉回 list 和一般數字, 才能直接存成 json
        f["pos"] = [float(pos[0]), float(pos[1])]
        f["yaw"] = float(yaw) % 360
    return blocker


def validate(rooms=None):
    """檢查所有家具都在所屬房間內, 且同房間家具彼此沒有重疊; 不合法就丟出 ValueError"""
    rooms = data["rooms"] if rooms is None else rooms
    for room in rooms:
        for f in room["furniture"]:
            dx, dy = clamp_shift(furniture_corners(room, f), room)
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                raise ValueError(f"{f['id']} is outside {room['id']}")
        items = room["furniture"]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if polygons_overlap(furniture_corners(room, items[i]), furniture_corners(room, items[j])):
                    raise ValueError(f"{items[i]['id']} overlaps {items[j]['id']} in {room['id']}")


def draw_environment():
    """顯示當前環境 (會跳出視窗顯示 matplotlib 繪製的環境，
        關閉視窗後才會繼續往下執行)"""
    fig, ax = plt.subplots(figsize=(11, 8))
    all_bounds = []

    for room in data["rooms"]:
        x0, y0, x1, y1 = room_bounds(room)
        all_bounds.append((x0, y0, x1, y1))
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='black', alpha=0.6))
        ax.text((x0 + x1) / 2, y1 - 0.15, room["id"], weight='bold', fontsize=11, ha='center', va='top')

        # 門: 在牆上畫一段粗線
        for door in room.get("doors", []):
            (xa, ya), (xb, yb) = door_segment(room, door)
            ax.plot([xa, xb], [ya, yb], color='saddlebrown', linewidth=6, solid_capstyle='butt', zorder=3)

        for f in room["furniture"]:
            corners = furniture_corners(room, f)
            ax.add_patch(patches.Polygon(
                corners, closed=True, linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.7))
            cx, cy = corners.mean(axis=0)
            ax.text(cx, cy, f"{f['id']}\n({room['id']})", fontsize=7, ha='center', va='center')

    ax.set_xlim(min(b[0] for b in all_bounds) - 0.5, max(b[2] for b in all_bounds) + 0.5)
    ax.set_ylim(min(b[1] for b in all_bounds) - 0.5, max(b[3] for b in all_bounds) + 0.5)
    ax.set_aspect('equal')
    plt.title("2D Room Floor Plan", fontsize=14, weight='bold')
    plt.xlabel("Width (m)")
    plt.ylabel("Length (m)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def move_furniture(f_id, dx, dy):
    """
    平移家具; 超出房間範圍就停留在邊界值。
    若移動後會與其他家具重疊, 則不移動, 並回傳擋住的家具 id (成功則回傳 None)。
    """
    room, f = locate(f_id)
    return place(room, f, np.array(f["pos"], dtype=float) + np.array([dx, dy]), f["yaw"])


def rotate_furniture(f_id, angle):
    """
    以家具中心旋轉 (pos 就是中心, 所以只要改 yaw); 超出房間範圍就推回邊界。
    若旋轉後會與其他家具重疊, 則不旋轉, 並回傳擋住的家具 id (成功則回傳 None)。
    """
    room, f = locate(f_id)
    return place(room, f, f["pos"], f["yaw"] + angle)


def mess_furniture(tries=50):
    """每件家具在所在房間內隨機旋轉並平移, 且不與其他家具重疊 (試 tries 次都不行就維持原位)"""
    for room in data["rooms"]:
        rx0, ry0, rx1, ry1 = room_bounds(room)
        for f in room["furniture"]:
            if f.get("fixed"):      # 固定的家具不打亂
                continue
            for _ in range(tries):
                # 隨機旋轉: 先算出轉到新角度後的頂點
                yaw = random.uniform(0, 360)
                corners = furniture_corners(room, f, yaw=yaw)
                x0, y0 = corners.min(axis=0)
                x1, y1 = corners.max(axis=0)
                # 隨機平移: 可移動範圍為家具到牆壁距離
                dx = random.uniform(rx0 - x0, rx1 - x1)
                dy = random.uniform(ry0 - y0, ry1 - y1)
                if place(room, f, np.array(f["pos"], dtype=float) + np.array([dx, dy]), yaw) is None:
                    break


def save_state(path=SAVE_FILE):
    """存成 .json (data 本來就是 list 和數字, 不需要 .tolist())"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read(path=SAVE_FILE):
    """
    讀取先前存檔的 .json, 還原到 data。
    存檔格式不對, 或有家具超出房間、互相重疊, 會丟出 ValueError, 不會改動目前狀態。
    """
    with open(path, "r", encoding="utf-8") as f:
        new_data = json.load(f)

    if not isinstance(new_data.get("rooms"), list):
        raise ValueError("Unrecognized save file format")
    validate(new_data["rooms"])

    # 原地更新 (所有函式用的都是同一個 data 字典)
    data.clear()
    data.update(new_data)


def ask_furniture():
    ids = [f["id"] for room in data["rooms"] for f in room["furniture"]]
    f_id = input(f"Enter furniture id {ids}: ")
    if f_id not in ids:
        raise ValueError(f"Unknown furniture: {f_id}")
    return f_id


def main():
    try:
        validate()   # info.json 的家具若超出房間或重疊, 先提醒 (不中斷程式)
    except ValueError as e:
        print(f"Warning: {e}")

    while True:
        draw_environment()          # 顯示當前環境
        cmd = input("Enter command (move/rotate/save/read/mess/quit): ").strip().lower()  # 跳出
        try:
            if cmd == "quit":
                break
            elif cmd == "move":
                f_id = ask_furniture()
                dx = float(input("Enter x offset (m): "))
                dy = float(input("Enter y offset (m): "))
                blocker = move_furniture(f_id, dx, dy)
                if blocker:
                    print(f"Blocked: {f_id} would overlap {blocker}, not moved.")
            elif cmd == "rotate":
                f_id = ask_furniture()
                angle = float(input("Enter rotation angle (degrees): "))
                blocker = rotate_furniture(f_id, angle)
                if blocker:
                    print(f"Blocked: {f_id} would overlap {blocker}, not rotated.")
            elif cmd == "save":
                save_state()
                print(f"Saved to {SAVE_FILE}")
            elif cmd == "mess":
                mess_furniture()
            elif cmd == "read":
                read()
                print(f"Loaded from {SAVE_FILE}")
            else:
                print("Unknown command.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
