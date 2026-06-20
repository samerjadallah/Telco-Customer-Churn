import pandas as pd
import numpy as np

# 1. قراءة ملف البيانات (تأكد من وجود الملف في نفس المجلد أو اكتب المسار الصحيح)
file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
try:
    df = pd.read_csv(file_path)
    print("✅ تم تحميل البيانات بنجاح!")
except FileNotFoundError:
    print("❌ لم يتم العثور على الملف. تأكد من المسار واسم الملف.")

# 2. معالجة عمود TotalCharges
# هذا العمود يحتوي أحياناً على فراغات نصية " " بسبب العملاء الجدد، سنحولها إلى NaN ثم نملأها بـ 0
df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# 3. تحويل المتغير المستهدف (Churn) إلى قيم رقمية
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 4. استعراض المعلومات الأساسية عن البيانات
print("\n📊 أبعاد البيانات (الصفوف، الأعمدة):", df.shape)
print("\nℹ️ معلومات عن الأعمدة وأنواعها:")
print(df.info())

# 5. عرض أول 5 صفوف
print("\n👀 نظرة على أول 5 صفوف من البيانات:")
print(df.head())

# 6. تفقد نسبة انفضاض العملاء (Churn Rate)
churn_rate = df['Churn'].mean() * 100
print(f"\n📉 نسبة انفضاض العملاء في الشركة: {churn_rate:.2f}%")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1. تحليل علاقة نوع العقد (Contract) بترك الخدمة (Churn)
plt.figure(figsize=(8, 5))
sns.barplot(x='Contract', y='Churn', data=df, ci=None, palette='viridis')
plt.title('Churn Rate by Contract Type')
plt.ylabel('Churn Rate (Percentage)')
plt.xlabel('Contract Type')
plt.show()

# 2. حذف العمود المعرفي الذي لا يحمل قيمة إحصائية للنموذج
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# 3. تحويل المتغيرات النصية الثنائية (Yes/No) إلى (1/0)
binary_cols = ['partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
# للتأكد من حالة الأحرف (Lower/Upper case) سنقوم بالتحويل الذكي
for col in df.columns:
    if df[col].dtype == 'object' and df[col].nunique() == 2:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

# 4. تحويل المتغيرات النصية متعددة الخيارات باستخدام One-Hot Encoding
# مثل (InternetService, PaymentMethod, Contract)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 5. تقسيم البيانات إلى متغيرات مستقلة (X) ومتغير مستهدف (y)
X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

# 6. تقسيم البيانات إلى مجموعتي التدريب والاختبار (80% تدريب، 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("✅ تم تجهيز البيانات وتقسيمها بنجاح!")
print(f"📐 حجم مجموعة التدريب (X_train): {X_train.shape}")
print(f"📐 حجم مجموعة الاختبار (X_test): {X_test.shape}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. بناء وتدريب نموذج الغابة العشوائية (Random Forest)
# استخدمنا class_weight='balanced' للتعامل الذكي مع عدم توازن البيانات
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10)
model.fit(X_train, y_train)

# 2. التنبؤ على مجموعة الاختبار
y_pred = model.predict(X_test)

# 3. تقييم أداء النموذج
print("📊 --- تقرير أداء النموذج (Classification Report) ---")
print(classification_report(y_test, y_pred))

print("📉 --- مصفوفة الارتباك (Confusion Matrix) ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# حساب الدقة الإجمالية
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 الدقة الإجمالية للنموذج (Accuracy): {accuracy * 100:.2f}%")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. استخراج الميزات وترتيبها حسب الأهمية
importances = model.feature_importances_
feature_names = X.columns

# إنشاء جدول يحتوي على الميزة ونسبة أهميتها
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10) # سنعرض أهم 10 ميزات

# 2. رسم الميزات بيانيّاً
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='magma')
plt.title('Top 10 Important Features in Predicting Customer Churn')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# 3. طباعة الميزات بشكل نصي مرتب
print("🏆 أهم 10 عوامل تؤثر على انفضاض العملاء:")
for index, row in feature_imp_df.iterrows():
    print(f"- {row['Feature']}: {row['Importance']*100:.2f}%")

