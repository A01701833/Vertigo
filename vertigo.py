# PAQUETES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas_profiling
from matplotlib import rcParams
import warnings


warnings.filterwarnings("ignore")


def vertigo(test_string):
    # figure size in inches
    rcParams["figure.figsize"] = 10, 6
    np.random.seed(42)

    data = pd.read_csv("vertigodata.csv")
    # IMPRIMIR DATOS CSV
    # print(data.sample(5))
    # Imprimir columnas
    # print (data.columns)

    eda_report = pandas_profiling.ProfileReport(data)
    #print(eda_report)
    # split data into input and taget variable(s)

    X = data.drop("AREA", axis=1)
    y = data["AREA"]
    # VARIABLES CATEGORICAS A NUMERICAS YA QUE SI NO APARECE UN ERROR DE CONVERSION
    X = pd.get_dummies(X, drop_first=True)

    y = pd.get_dummies(y, prefix="", prefix_sep="").max(level=0, axis=1)
    # y = pd.get_dummies(y, drop_first=True)

    """print("COLUMNAS X  ")
    print(X.columns.values)
    print("COLUMNAS Y ")
    print(X)"""
    # standardize the dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DIVIDIR EN EL SET
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.10, random_state=42
    )

    # CREAR EL BOSQUE ALEATORIO
    classifier = RandomForestClassifier(n_estimators=100)

    # ENTRENAR EL MODELO
    classifier.fit(X_train, y_train)
    # PREDECIR EL CONJUNTO DE DATOS
    y_pred = classifier.predict(X_test)
    # PRECISION ( TIENE QUE SER MAYOR A 90 )
    #print("Precisión:", accuracy_score(y_test, y_pred))




    feature_importances_df = pd.DataFrame(
        {"caracteristica": list(X.columns), "importancia": classifier.feature_importances_}
    ).sort_values("importancia", ascending=False)
    # CARACTERISTICAS IMPORTANTES
    # Display
    #print(feature_importances_df)

    # matrix

    matrix = confusion_matrix(
        y_test.values.argmax(axis=1), y_pred.argmax(axis=1))

    #print(matrix)
    df_cm = pd.DataFrame(matrix, range(4), range(4))
    # plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    #plt.show()

    """
    ['VOMITO_SI' 'DURACION_HORAS' 'DURACION_MINUTOS' 'DURACION_SEGUNDOS'
     'EPISODIO_PROGRESIVO' 'FRECUENCIA_RECURRENTE' 'FRECUENCIA_UNA VEZ'
     'MEDICAMENTO_SI' 'PALIDEZ_SI' 'SUDORACION_SI' 'ARRITMIAS_SI' 'DESMAYO_SI'
     'VISION_SI' 'PRESION_SI' 'TRAUMA_SI' 'ESTOMAGO_SI' 'HIPOACUSIA_SI'
     'MOVILIDAD_SI' 'OXIGENACION_SI']
     """
    # NUEVA PREDICCION AQUI ENTRA EL INPUT
    # SHOULD OUTPUT CARDIOLOGIA [ 1 0 0 0 ]
    cardio = [[1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]]
    # SHOULD OUTPUT NEUROLOGIA [ 0 1 0 0 ]
    neuro = [[1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]]

    # SHOULD OUTPUT OTORRINOLARINGOLOGIA [ 0 0 1 0 ]
    otorrino = [[1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]]
    # SHOULD OUTPUT URGENCIAS [ 0 0 0 1 ]
    urgen = [[1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]]

    neww = classifier.predict(test_string)
    #neww = classifier.predict(urgen)

    finalstr = "EL AREA DE ATENCION A LA QUE DEBES ACUDIR ES: "
    if str(neww[0]) == "[1 0 0 0]":
        print(finalstr + " CARDIOLOGIA")
    elif str(neww[0])  == "[0 1 0 0]":
        print(finalstr + " NEUROLOGIA")
    elif str(neww[0]) == "[0 0 1 0]":
        print(finalstr + " OTORRINOLARINGOLOGIA")
    else:
        print(finalstr + " URGENCIAS")

    # MATRIZ DE CONFUSION

    # PORCENTAJE DE CADA UNO
    # gitter para un dataset.


    # FIN DE LA PREDICCION

    # VISUALIZAR

    # GRAFICA DE BARRAS
    sns.barplot(x=feature_importances_df.caracteristica, y=feature_importances_df.importancia)
    # LABELS
    plt.xlabel("RESULTADO DE CARACTERÍSTICAS IMPORTANTES")
    plt.ylabel("CARACTERÍSTICA")
    plt.title("VISUALIZACIÓN DE CARACTERISTICAS IMPORTANTES")
    plt.xticks(
        rotation=45, horizontalalignment="right", fontweight="light", fontsize="x-large"
    )
    #plt.show()


if __name__ == "__main__":

    """
    ['VOMITO_SI' 'DURACION_HORAS' 'DURACION_MINUTOS' 'DURACION_SEGUNDOS'
     'EPISODIO_PROGRESIVO' 'FRECUENCIA_RECURRENTE' 'FRECUENCIA_UNA VEZ'
     'MEDICAMENTO_SI' 'PALIDEZ_SI' 'SUDORACION_SI' 'ARRITMIAS_SI' 'DESMAYO_SI'
     'VISION_SI' 'PRESION_SI' 'TRAUMA_SI' 'ESTOMAGO_SI' 'HIPOACUSIA_SI'
     'MOVILIDAD_SI' 'OXIGENACION_SI']
     EX : cardio = [[1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]]
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]

"""
    preguntas = ["¿Tienes Vomito? SI (1) NO (0) :  ",
                 "¿Duración del episodio?  HORAS (1) MINUTOS (2) SEGUNDOS (3) :  ",
                 "Tipo de Episodio -  PROGRESIVO (1) BRUSCO (0) : ",
                 "¿Que tan frecuente es el episodio? :  SOLO UNA VEZ (1) - A VECES (2) - SEGUIDO (3) : ",
                 "¿Has tomado algun medicamento en las ultimas 24 horas? : SI (1) NO (0) : ",
                 "¿Tienes Palidez? (notas tu piel un poco mas blanca-amarillosa) SI (1) NO (0) : ",
                 "¿Estas sudando? SI (1) NO (0) : ",
                 "¿Sientes raro como late tu corazon? SI (1) NO (0) : ",
                 "¿Sientes que te estas desmayando?  SI (1) NO (0) :  ",
                 "¿Tienes la vision borrosa? SI (1) NO (0) : ",
                 "¿Tienes historial de presion arterial baja? SI (1) NO (0) : ",
                 "¿Tuviste algun golpe en la cabeza en las ultimas 24 horas? SI (1) NO (0) : ",
                 "¿Tienes dolor en el estomago? SI (1) NO (0) : ",
                 "¿Sientes que escuchas un poco menos o alguna otra dificultad en el oido?  SI (1) NO (0) : ",
                 "¿Tienes algun problema para moverte? SI (1) NO (0) : ",
                 "¿Tienes problemas para respirar? SI (1) NO (0) : "
                 ]
    test = []
    
    flag = False
    for x in range(len(preguntas)):
        try:
            if flag:
                aux = int(input(preguntas[x - 1]))
                flag = False
            else:
                aux = int(input(preguntas[x]))
            if len(str(aux)) == 1:
                if x == 1:
                    if aux == 1:
                        test.extend([1, 0, 0])
                    elif aux == 2:
                        test.extend([0, 1, 0])
                    else:
                        test.extend([0, 0, 1])
                elif x == 3:
                    if aux == 1:
                        test.extend([0, 1])
                    elif aux == 2:
                        test.extend([0, 0])
                    else:
                        test.extend([1, 0])
                else:
                    test.append(aux)
            else:
                flag = True
        except ValueError:
            # Handle the exception
            flag = True
            print(test)
            print('Por favor solo ingrese numeros disponibles. ')

    test2d = []
    test2d.append(test)

    vertigo(test2d)
