import pandas as pd
import numpy as np
# groupby()多标签归类
df1 = pd.DataFrame({"col1":list("ababbc"),
                   "col2":list("xxyyzz"),
                   "number1":range(90,96),
                   "number2":range(100,106)})
print(df1)
df2 = df1.groupby(["col1","col2"]).agg({"number1":sum,
                                        "number2":np.mean})

print(df2)