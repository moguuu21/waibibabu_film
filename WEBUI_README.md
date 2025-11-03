# PyCinemetrics WebUI (鏈湴鐗?

杩愯涓€涓湰鍦?Web UI锛岄€氳繃 HTTP API 鍦?localhost 涓婅皟鐢ㄧ幇鏈夌畻娉曘€?
## 蹇€熷紑濮?(Windows)

- 瀹夎鏈嶅姟鍣ㄤ緷璧栵細
  - `pip install -r webui/requirements.txt`
- 鍚姩鏈嶅姟鍣細
  - 鍙屽嚮 `run_webui.bat` 鎴栧湪缁堢涓繍琛?`.\run_webui.bat`
- 鍦ㄦ祻瑙堝櫒涓墦寮€锛歚http://127.0.0.1:8000`

## 浣跨敤鏂规硶

- 鏂瑰紡涓€锛氳緭鍏ュ畬鏁寸殑鏈湴瑙嗛璺緞锛屼緥濡?`C:\\videos\\movie.mp4`
- 鏂瑰紡浜岋細鐩存帴灏嗚棰戞枃浠舵嫋鎷藉埌杈撳叆妗嗕笅鏂圭殑鈥滄嫋鎷戒笂浼犫€濆尯鍩燂紝鎴栫偣鍑昏鍖哄煙閫夋嫨鏂囦欢銆備笂浼犲畬鎴愬悗浼氳嚜鍔ㄥ～鍏呰矾寰勩€?- 鐐瑰嚮鍏朵腑涓€涓搷浣滐細Shotcut锛堥暅澶村垏鎹級銆丆olors锛堣壊褰╋級銆丱bjects锛堢墿浣擄級銆丼ubtitles锛堝瓧骞曪級銆丼hotScale锛堥暅澶存櫙鍒級
- 缁撴灉淇濆瓨鍦?`img/<video_basename>/` 鐩綍涓嬶紝骞跺湪 UI 涓瑙?
## API 绔偣

- `POST /api/shotcut` JSON `{ video_path, th? }`
- `POST /api/colors` JSON `{ video_path, colors_count? }`
- `POST /api/objects` JSON `{ video_path }`
- `POST /api/subtitles` JSON `{ video_path, subtitle_value? }`
- `POST /api/shotscale` JSON `{ video_path }`
- `GET  /api/results?video_path=...` -> 鍒楀嚭鍙敤鐨勭粨鏋滃拰濯掍綋 URL

## 閰嶇疆

- 涓绘満/绔彛锛氳缃幆澧冨彉閲?`WEB_HOST`銆乣WEB_PORT`锛堥粯璁や负 `127.0.0.1:8000`锛?
## 娉ㄦ剰浜嬮」

- 鏈嶅姟鍣ㄩ噸鐢?`src/algorithms` 涓殑妯″潡銆傚浜庨渶瑕佸抚鐨勪换鍔★紝濡傛灉缂哄皯鍦烘櫙鍏抽敭甯э紝灏嗛€氳繃 TransNetV2 鎻愬彇銆?- 鐗╀綋妫€娴嬮渶瑕?`models/` 鐩綍涓嬬殑 YOLO 鏂囦欢锛堟湰浠撳簱涓凡鍖呭惈锛夈€?- 鎵€鏈夊鐞嗛兘鍦ㄦ湰鍦拌繘琛岋紱鏈嶅姟鍣ㄤ笉鎻愪緵杩滅▼璁块棶銆?
