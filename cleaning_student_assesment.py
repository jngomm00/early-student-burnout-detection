import pandas as pd

df_st_assess = pd.read_csv('./data/studentAssessment.csv')

# Solo nos interesan entregas realizadas hasta el día 90
df_st_assess_90 = df_st_assess[df_st_assess['date_submitted'] <= 90]

df_st_assess_90.to_csv('./data-clean/studentAssessment_90days.csv', index=False)
print("Paso 3 completado: Notas filtradas a 90 días.")