# Label Tool 操作說明

## 啟動

1. 執行 `label_tool.exe`。
2. 程式開啟後，按左側的 `Open Video`。
3. 選擇要標記的影片。

## 標記方式

- 左鍵點影片中的球位置：標記目前 frame 的球座標。
- 右鍵點影片中的球位置：同樣會標記目前 frame 的球座標。
- 若標錯目前 frame：按 `c` 清除。
- 若要微調位置：
  - `i`：往上 1 px
  - `j`：往左 1 px
  - `k`：往下 1 px
  - `l`：往右 1 px

## 切換 Frame

- `x`：下一 frame
- `z`：上一 frame
- `f`：往後 50 frames
- `d`：往前 50 frames
- 空白鍵：播放 / 暫停
- 下方 slider：快速跳到指定 frame

## 放大與移動畫面

- 滑鼠滾輪或右側 `Zoom` 拉桿：放大 / 縮小影片。
- 放大後，影片區會出現水平 / 垂直拉桿。
- 使用拉桿可以移動到畫面的左右或上下位置。

## 事件標記

目前支援事件值：

- `0`
- `1`
- `2`
- `3`

按鍵 `0` 到 `3` 會設定目前 frame 的 `Event` 欄位。

## 儲存

- 按 `s` 儲存 CSV。
- 按 `q` 會先儲存，再關閉程式。
- 關閉視窗時，程式會詢問是否儲存。

CSV 會自動輸出到：

- 如果影片在 `video` 資料夾內：輸出到同層的 `csv` 資料夾。
  - 範例：`project/video/a.mp4`
  - 輸出：`project/csv/a_ball.csv`
- 如果影片不在 `video` 資料夾內：輸出到影片所在資料夾底下的 `csv` 資料夾。
  - 範例：`D:/data/a.mp4`
  - 輸出：`D:/data/csv/a_ball.csv`

## CSV 欄位

輸出的 CSV 欄位如下：

```text
Frame,Visibility,X,Y,Z,Event,Timestamp,Fast
```

欄位意義：

- `Frame`：frame 編號。
- `Visibility`：是否有標記，`1` 代表有球，`0` 代表無球。
- `X` / `Y`：球座標。
- `Z`：保留欄位，目前預設 `0`。
- `Event`：事件值。
- `Timestamp`：時間秒數。
- `Fast`：保留欄位，目前預設 `0`。

## 換影片

不需要重開程式。直接按左側 `Open Video` 選新影片即可。

切換影片前，程式會詢問是否儲存目前標記。
