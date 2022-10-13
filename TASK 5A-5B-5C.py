"""
TASK 5B

Gözetimli Öğrenme
1) Regresyon örneği
Basit doğrusal regresyon örneği olarak elimizde reklama harcanan bütçe ve satış sayısı verileri varken ne kadar reklam bütçesiyle ne kadar sayısı yapılabileceği tahmini

2) Sınıflandırma Örneği
4 özelliğini(ram,cpu,gpu, hard disk tipi) ve labellerını(günlük kullanım, oyun ve workstation) bildiğimiz bir bilgisayar donanım özellikleri veri setimiz olsun. Bunlara dayanarak bize verilen 4 donanım bilgisinden bilgisayarın günlük kullanım için mi yoksa oyun bilgisayarı mı yada bir workstation mı olduğu tahmini 

Gözetimsiz Öğrenme
1) Kümeleme örneği
Bir raftaki ürün sayısını bulmak için fotoğrafı çekilen rafın opencv ile gerekli dezenformasyonlar uygulanarak kümeledikten sonra sayımı 

2) Boyut azaltma(dimensionality reduction) örneği
kanser tespiti için kullanılan 1000 boyutlu bir veri setinden önemli olduğu gözlenen 3 özniteliğe indirgeme örneği verilebilir.
"""

#TASK 5C
from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape
import matplotlib.pyplot as plt

X = digits.data
y = digits.target

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

score = []
rank = []
plt.figure()
for i in range(0,20):
    value = '1e-'+str(i)
    model = GaussianNB(var_smoothing=float(value))
    model.fit(Xtrain, ytrain)
    y_model = model.predict(Xtest)
    accscore = accuracy_score(ytest, y_model)
    score.append(accscore)
    rank.append(value)
    plt.scatter(value,accscore)
    print("var_smoothing=",value,"için score:",accscore)
plt.plot(rank,score)
plt.xticks(rotation=90)
plt.show()
print("Naive Bayes algoritması ile maksimum doğruluk değeri: var_smoothing=",rank[score.index(max(score))]," değeri ile", max(score),"olarak elde edilmiştir")

# TAKS 5D
# 1)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(Xtrain, ytrain)
y_model = neigh.predict(Xtest)
print("KNN algoritması ile doğruluk oranı: ", accuracy_score(ytest, y_model))

# 2)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(Xtrain, ytrain)
y_model = clf.predict(Xtest)
print("Random Forest algoritması ile doğruluk oranı: ", accuracy_score(ytest, y_model))













