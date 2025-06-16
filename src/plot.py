# tensorboard csv file plot code

import pandas as pd
import matplotlib.pyplot as plt

# 1) CSV 파일 경로 (plot.py가 있는 디렉토리 기준)
train_csv = '../csv/train_acc.csv'
val_csv   = '../csv/val_acc.csv'

# 2) 두 번째 줄(header=1)을 컬럼명으로 읽기
train_df = pd.read_csv(train_csv, header=1)
val_df   = pd.read_csv(val_csv,   header=1)

# 3) step/value 꺼내기 (2번째 열, 3번째 열)
train_steps = train_df.iloc[:, 1]
train_vals  = train_df.iloc[:, 2]

val_steps   = val_df.iloc[:, 1]
val_vals    = val_df.iloc[:, 2]

# 4) 그리기
plt.figure(figsize=(8, 5))
plt.plot(train_steps, train_vals, color='blue', label='train')
plt.plot(val_steps,   val_vals,   color='red', label='validation')

plt.xlabel('step')
plt.ylabel('value')
plt.title('Epoch Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()
