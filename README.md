# 2 Iris Flower Prediction App


1. [Importar librerías ](#schema1)
2. [Título](#schema2)
3. [Título de barra lateral](#schema3)
4. [Subtítulo](#schema4)
5. [Función que recoge los datos de la barra lateral](#schema5)
6. [Imprimir por pantalla el dataframe obtenido](#schema6)
7. [Modelo, entrenarlo y predecir](#schema7)
8. [Imprimir `target_names` de lo datos cargados `iris`](#schema8)
9. [Imprimir la predicción con los datos de usuario](#schema9)
10. [Imprimir la probabilidad de la predicción](#schema10)
11. [Documentación](#schema11)

<hr>

<a name="schema1"></a>
 
# 1. Importar librerías
~~~python
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
~~~

<hr>

<a name="schema2"></a>

# 2. Título
~~~python
st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")
~~~

<hr>

<a name="schema3"></a>

# 3. Título de barra lateral
~~~python
st.sidebar.header('User Input Parameters')
~~~
<hr>

<a name="schema4"></a>

# 4. Subtítulo
~~~python
st.subheader('User Input parameters')
~~~

<hr>

<a name="schema5"></a>

# 5. Función que recoge los datos de la barra lateral

~~~python
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
~~~
~~~python
streamlit.slider(label, min_value, max_value, value)
~~~
value = valor actual

<hr>

<a name="schema6"></a>

# 6. Imprimir por pantalla el dataframe obtenido

~~~Python
st.write(df)
~~~
<hr>

<a name="schema7"></a>

# 7. Modelo, entrenarlo y predecir
~~~python
iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)
~~~
<hr>

<a name="schema8"></a>

# 8. Imprimir `target_names` de lo datos cargados `iris`
~~~python
st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)
~~~
<hr>

<a name="schema9"></a>

# 9. Imprimir la predicción con los datos de usuario
~~~python
st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)
~~~

<hr>

<a name="schema10"></a>

# 10. Imprimir la probabilidad de la predicción
~~~python
st.subheader('Prediction Probability')
st.write(prediction_proba)

~~~

# 11. Documentación
https://www.youtube.com/watch?v=8M20LyCZDOY&list=PLtqF5YXg7GLmCvTswG32NqQypOuYkPRUE&index=2

https://github.com/dataprofessor/code/blob/master/streamlit/part2/iris-ml-app.py
