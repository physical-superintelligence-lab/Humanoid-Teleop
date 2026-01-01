import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("tau_log_1767159696.csv") # 替换成你的文件名

# 绘图
plt.figure(figsize=(10, 5))
for col in df.columns[1:]:  # 跳过时间列 't'
    plt.plot(df['t'], df[col], label=col)

plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.title('Last 7 Dimensions of TauState for Carrying Box Task')
plt.legend()
plt.grid(True)
plt.show()