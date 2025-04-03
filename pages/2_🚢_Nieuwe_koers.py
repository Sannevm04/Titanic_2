import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from streamlit_option_menu import option_menu
import plotly.subplots as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# opmaak
st.set_page_config(
    page_title="Nieuwe koers",
    page_icon="🚢",
    layout="wide"
)
st.title("🚢 Verbeterpoging")
with st.sidebar: 
    pagina = option_menu('Inhoudsopgave', ["Data verkenning", "Analyse", "Voorspellend model"], icons=['search', 'bar-chart-line', 'graph-up-arrow'], menu_icon='card-list')

# data inladen
@st.cache_data
def load_train_new():
    train_new = pd.read_csv('train.csv')
    train_new.drop(columns = ['Ticket', 'Cabin', 'PassengerId'], 
                   axis = 1, inplace = True)
    
    # nieuwe kolommen    # leeftijdscategorieën toevoegen
    def leeftijdscategorie(leeftijd):
        if leeftijd < 16:
            return 1
        elif 16 <= leeftijd <= 34:
            return 2
        elif 35 <= leeftijd <= 49:
            return 3
        elif 50 <= leeftijd <= 64:
            return 4
        else:
            return 5
    df= pd.DataFrame({'Leeftijd': train_new['Age']})
    train_new['Age_categories'] = df['Leeftijd'].apply(leeftijdscategorie)

    # travel alone, of met gezelschap
    train_new['Travel_budy'] = train_new['SibSp'] + train_new['Parch'] + 1
    train_new['IsAlone'] = (train_new['Travel_budy'] == 1).astype(int)
    train_new['Family'] = (train_new['Parch'] + train_new['SibSp']).apply(lambda x: 1 if x > 0 else 0)
    train_new['Small_fam'] = (train_new['Parch'] + train_new['SibSp'] + 1).apply(lambda x: 1 if x < 5 else 0)

    # Naam naar titels geven
    train_new['Title'] = train_new['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    train_new['Title'] = train_new['Title'].map(lambda x: title_mapping.get(x, 5))
    
    # locaties
    train_new['Embarked'] = train_new['Embarked'].replace({'S':'Southampthon', 'C':'Cherbourgh', 'Q': 'Queenstown'})

    # leeftijd en geslacht
    def child_female_male(passenger):
        Age, Sex = passenger
        if Age < 16:
            return 'child'
        else:
            return Sex
    train_new['Type'] = train_new[['Age', 'Sex']].apply(child_female_male, axis = 1)

    # fare
    train_new['Fare']= train_new['Fare'].astype(int)
    return train_new

@st.cache_data
def load_test_new():
    test_df = pd.read_csv("test.csv")

    # leeftijd
    test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
    def leeftijdscategorie(leeftijd):
        if leeftijd < 16:
            return 1
        elif 16 <= leeftijd <= 34:
            return 2
        elif 35 <= leeftijd <= 49:
            return 3
        elif 50 <= leeftijd <= 64:
            return 4
        else:
            return 5
    df= pd.DataFrame({'Leeftijd': test_df['Age']})
    test_df['Age_categories'] = test_df['Age'].apply(leeftijdscategorie)

    #reisgenoten
    test_df['Travel_budy'] =test_df['SibSp'] + test_df['Parch'] + 1
    test_df['IsAlone'] = (test_df['Travel_budy'] == 1).astype(int)
    test_df['Family'] = (test_df['Parch'] + test_df['SibSp']).apply(lambda x: 1 if x > 0 else 0)
    test_df['Small_fam'] = (test_df['Parch'] + test_df['SibSp'] + 1).apply(lambda x: 1 if x < 5 else 0)

    # titels
    test_df['Title'] = test_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    test_df['Title'] = test_df['Title'].map(lambda x: title_mapping.get(x, 5))
   
    # locaties
    test_df['Embarked'].fillna('S', inplace=True)
    test_df['Embarked'] = test_df['Embarked'].replace({'S':'Southampthon', 'C':'Cherbourgh', 'Q': 'Queenstown'})

    # Faire
    test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
    test_df['Fare'] = test_df['Fare'].astype(int)

    # leeftijd en geslacht
    def child_female_male(passenger):
        Age, Sex = passenger
        if Age < 16:
            return 'child'
        else:
            return Sex
    test_df['Type'] = test_df[['Age', 'Sex']].apply(child_female_male, axis = 1)
    return test_df

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, 'Rare':5, "Sir":1,"Lady":3}
title_demapping = {1: "Mr", 2: "Miss", 3: "Mrs", 4: "Master", 5: "Rare"}
sex_mapping = {"male":0, "female":1}
sex_demapping = {0:'male', 1:"female"}
embarked_mapping = {'Southampthon':0, 'Cherbourgh':1,'Queenstown':2}
embarked_demapping = {0:'Southampthon', 1:'Cherbourgh', 2:'Queenstown'}
Pclass_mapping = {'1e klas': 1, '2e klas':2, '3e klas':3}
Pclass_demapping = {1: '1e klas', 2:'2e klas', 3:'3e klas'}
Alone_mapping = {'Alleen':1, 'Samen':0}
Alone_demapping = {1:'Alleen', 0:'Samen'}
Age_cat_mapping = {'Kind (<16)':1, 'Jong volwassen (16<=34)':2, 'Volwassen (35<=49)':3, 'Middelbare leeftijd (50<=64)':4, 'Oudere (>65)':5}
Age_cat_demapping = {1: 'Kind (<16)', 2:'Jong volwassen (16<=34)', 3:'Volwassen (35<=49)', 4:'Middelbare leeftijd (50<=64)', 5:'Oudere (>65)'}
type_mapping = {'male':1,'female':2,'child':3}
type_demapping = {1:'male', 2:'female',3:'child'}

train_new = load_train_new()
test_df = load_test_new()

if pagina == 'Data verkenning':
    st.header('1. Data verkenning')
    st.dataframe(train_new.head())
    kolommen = train_new.columns

    st.subheader('1.1 Data opvulling')
    st.write('De missende waardes komen voor in leeftijd en Opstaplocatie:')
    missing_values = train_new.isnull().sum()
    missing_values = missing_values[missing_values>0].reset_index()
    missing_values.columns = ['Kolom', 'Aantal missende waardes']
    st.dataframe(missing_values, hide_index=True, use_container_width=True)
    st.info(
        f"In totaal zijn er {missing_values['Aantal missende waardes'].sum()} missende waardes "
        f"verdeeld over {missing_values.shape[0]} kolommen. Deze waardes gaan we dan opvullen."
    )

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Verdeling Leeftijd", "Opstaplocatie"])
    fig.add_trace(
        go.Histogram(x=train_new["Age"], nbinsx=10, marker_color="lightblue", name="Leeftijd"), 
        row=1, col=1)

    # Voeg histogram voor ticketprijs toe
    fig.add_trace(
        go.Histogram(x=train_new["Embarked"], nbinsx=10, marker_color="lightgreen", name="Ticketprijs"), 
        row=1, col=2)

    # Pas de layout aan
    fig.update_layout(
        title_text="Histogrammen van Leeftijd en Opstaplocatie",
        showlegend=False,
        height=500, width=900    )

    # Toon in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"De leeftijd heeft een mediaan ({train_new['Age'].median()}) die bijna gelijk is aan het gemiddelde ({np.round(train_new['Age'].mean())})" 
        "en zal daarom opgevuld worden met de mediaan  \n"
        f"De opstaplocatie heeft bij Southampthon zo'n hoge waarde, dat deze gebruikt wordt om de missende waardes in te vullen"
    )
    train_new['Age'].fillna(train_new['Age'].median(), inplace=True)
    train_new['Embarked'].fillna('S', inplace=True)

    st.subheader('1.2 Nieuwe waardes toevoegen')
    st.write("Naast de basis kolommen, zijn er ook kolommen toegevoegd:  \n"
    "* Leeftijd is onderverdeeld in categoriën  \n"
    "* Reisgenoten is toegevoegd, om aan te geven om iemand alleen of samen reisde  \n"
    "* Voorvoegsel van de naam is achterhaald")

    # Nieuwe data kolommen
    st.dataframe(train_new[['Age','Age_categories','Travel_budy', 'IsAlone','Title', 'Family','Type','Small_fam']].head())

    st.subheader('1.3 correlatie matrix')
    train_new['Type'] = train_new['Type'].map(type_mapping)
    train_new['Embarked'] = train_new['Embarked'].map(embarked_mapping)
    correlatie_matrix = train_new.drop(columns=['Name','Sex']).corr()
    correlatie_matrix['abs'] = correlatie_matrix['Survived'].abs()
    def classificatie(r):
        if abs(r) > 0.5:
            return "Sterk"
        elif abs(r) > 0.3:
            return "Matig"
        elif abs(r) > 0:
            return "Zwak"
        else:
            return "Geen"
    correlatie_matrix['Sterkte'] = correlatie_matrix['Survived'].apply(classificatie)
    correlatie_matrix = correlatie_matrix.drop('Survived').sort_values(by='abs', ascending=False).drop(columns=['abs'])
    
    # Maak aangepaste annotaties met zowel de correlatie als de classificatie
    annotaties = correlatie_matrix.apply(lambda row: f"{row['Survived']:.2f}\n({row['Sterkte']})", axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        correlatie_matrix[['Survived']], 
        annot=annotaties.values.reshape(-1,1),  # Gebruik aangepaste annotaties
        cmap="coolwarm", fmt="", linewidths=0.5, center=0, ax=ax
    )
    ax.set_title("Correlatie over de overlevingskans op de Titanic")
    # Display plot in Streamlit
    st.pyplot(fig)
    st.info("Het lijkt interessant te zijn om te kijken naar:  \n* Type (geslacht en kind)  \n* Title  \n* Klas  "
    "\n* Ticketprijs   \n* Familie  \n* Alleen of samen reizen  \n* Leeftijd (in groepen) \n" \
    "* Groote van de familie \n* Opstaplocatie")

    st.subheader("1.4 Verdieping in de variabelen")
    st.write("Om meer inzicht te krijgen in de variabelen kan hieronder gekozen worden voor variabelen waar meer verdiept in kan worden")
    Variabele = st.selectbox("Selecteer een variabele",options=['Type','Title','Pclass','Fare','IsAlone','Age_categories','Embarked','Small_fam'])
    
    st.markdown("##### 1.4.1 Histogram")
    train_new['Type'] = train_new['Type'].map(type_demapping)
    train_new['Embarked'] = train_new['Embarked'].map(embarked_demapping)
    train_new['Title'] = train_new['Title'].map(title_demapping)
    train_new['Pclass'] = train_new['Pclass'].map(Pclass_demapping)
    train_new['IsAlone'] = train_new['IsAlone'].map(Alone_demapping)
    train_new['Age_categories'] = train_new['Age_categories'].map(Age_cat_demapping)

    if Variabele == "Type":
        kleur_dict = {
        "male": "lightblue",  
        "female": "pink",
        "child": "lightgreen"}
        fig = px.histogram(train_new, x=Variabele, title=f"Histogram van {Variabele}",
                        color=train_new['Sex'],
                        color_discrete_map=kleur_dict)
        st.plotly_chart(fig)
    else:
        fig = px.histogram(train_new, x=Variabele, title=f"Histogram van {Variabele}", color_discrete_sequence=['lightgreen'])
        st.plotly_chart(fig)
    
    
    
    st.markdown("##### 1.4.2 Onderlinge correlatie")
    train_new['Type'] = train_new['Type'].map(type_mapping)
    train_new['Embarked'] = train_new['Embarked'].map(embarked_mapping)
    train_new['Title'] = train_new['Title'].map(title_mapping)
    train_new['Pclass'] = train_new['Pclass'].map(Pclass_mapping)
    train_new['IsAlone'] = train_new['IsAlone'].map(Alone_mapping)
    train_new['Age_categories'] = train_new['Age_categories'].map(Age_cat_mapping)

    correlatie_matrix_Var = train_new.drop(columns=['Name','Age','SibSp','Parch','Sex',"Family"]).corr()
    correlatie_matrix_Var['abs'] = correlatie_matrix_Var[Variabele].abs()
    correlatie_matrix_Var['Sterkte'] = correlatie_matrix_Var[Variabele].apply(classificatie)
    correlatie_matrix_Var = correlatie_matrix_Var.sort_values(by='abs', ascending=False).drop(columns=['abs']).head(3)
    
    # Maak aangepaste annotaties met zowel de correlatie als de classificatie
    annotaties = correlatie_matrix_Var.apply(lambda row: f"{row[Variabele]:.2f}\n({row['Sterkte']})", axis=1)
    fig, ax = plt.subplots(figsize=(5, 2))
    sns.heatmap(
        correlatie_matrix_Var[[Variabele]], 
        annot=annotaties.values.reshape(-1,1),  # Gebruik aangepaste annotaties
        cmap="coolwarm", fmt="", linewidths=0.5, center=0, ax=ax
    )
    ax.set_title("Onderling Correlatie")
    # Display plot in Streamlit
    st.pyplot(fig)

    if Variabele == "Sex":
        st.info("Er is een correlatie met Titel. Dit is ook zeker logisch, "
        "gezien een man over het algemeen met Mr/Master aangesproken, en een vrouw met Mrs/Miss."
        "Toch blijft Sex en Title voor nu van belang om te bepalen of bepaalde Titels meer/minder overlevingskans hebben.")
    elif Variabele == 'Title':
        st.info("Er is een correlatie met geslacht. Dit is ook zeker logisch, "
        "gezien een man over het algemeen met Mr/Master aangesproken, en een vrouw met Mrs/Miss."
        "Toch blijft Sex en Title voor nu van belang om te bepalen of bepaalde Titels meer/minder overlevingskans hebben.")
    elif Variabele == 'Pclass':
        st.info("Er is een correlatie met Ticketprijs. Dit is ook wel te verwachten, "
        "gezien dat met de klasse stijging ook de prijs mee zal stijgen.")
    elif Variabele == 'Fare':
        st.info("Er is een correlatie met Klasse. Dit is ook wel te verwachten, "
        "gezien dat met de klasse stijging ook de prijs mee zal stijgen.")
    elif Variabele == 'IsAlone':
        st.info('Alleen reizen correleerd met of je reisgenoten hebt, dit is logisch. Beide zullen onafhankelijk nog woren meegenomen.')
    
    if Variabele == 'Age_categories':
        st.markdown('##### Andere handige visualisaties')
        fig = px.box(train_new, y='Age')
        st.plotly_chart(fig)
    elif Variabele == 'Fare':
        st.markdown('##### Andere handige visualisaties')
        fig = px.box(train_new, y=Variabele)
        st.plotly_chart(fig)

elif pagina == 'Analyse':
    st.header('2. Analyse')
    st.write('De onafhankelijke variabelen zullen hier verder onderzocht worden tegenover de afhankelijke varariabele: overleefingskans.')
    train_new['Age'].fillna(train_new['Age'].median(), inplace=True)
    train_new['Embarked'].fillna('S', inplace=True)

    st.subheader('2.1 Individuele variabelen tegenover overlevingskans')
    #demapping
    train_new['Title'] = train_new['Title'].map(title_demapping)
    train_new['Age_categories']= train_new['Age_categories'].map(Age_cat_demapping)
    train_new['IsAlone'] = train_new['IsAlone'].map(Alone_demapping)
    train_new["Pclass"] = train_new["Pclass"].map(Pclass_demapping)

    # Functie om overlevingskans per categorie te berekenen en te plotten
    def plot_multiple_graphs(data, variables, target='Survived'):
        rows, cols = 4, 2
        fig = sp.make_subplots(rows=rows, cols=cols, subplot_titles=variables)

        for i, variable in enumerate(variables):
            # Bereken de overlevingskans per variabele
            survival_by_var = data.groupby([variable])[target].mean() * 100
            
            # Maak een bar chart
            trace = go.Bar(
                x=survival_by_var.index,
                y=survival_by_var.values,
                text=np.round(survival_by_var.values, 2),  # Percentage met 2 decimalen als tekst
                hoverinfo='x+y+text',
                name=f'Overlevingskans per {variable}'
            )

            # Voeg de trace toe aan de subplot
            fig.add_trace(trace, row=(i // cols) + 1, col=(i % cols) + 1)

            # Update de y-as limieten voor elke subplot
            fig.update_yaxes(range=[0, 100], row=(i // cols) + 1, col=(i % cols) + 1)

        # Layout aanpassen
        fig.update_layout(
            title="Overlevingskans per categorieën",
            showlegend=False,
            height=800,  # Aangepaste hoogte voor de subplots
            yaxis=dict(range=[0, 100], title='Overlevingskans (%)')
        )

        # Toon de grafiek in Streamlit
        st.plotly_chart(fig)

    # Selecteer welke variabelen je wilt plotten (bijvoorbeeld 'Pclass', 'Age', 'Sex')
    variables_to_plot = st.multiselect('Kies de variabelen om te plotten:', ['Type','Title','Age_categories','IsAlone','Embarked','Pclass','Family','Small_fam'], 
                                       default=['Type','Title','Age_categories','IsAlone','Embarked','Pclass',"Family", 'Small_fam'])

    # Grafieken tonen
    plot_multiple_graphs(train_new, variables_to_plot)

    st.info("De conclusies uit de grafieken, als alles bekeken wordt:  \n"
    "* Als vrouw heb je een hoge overlevingskans, als man was je overlevingskans erg laag.  \n"
    "* Kinderen (<16) hebben een overlevingskans van 60%, alles ouder overleed meer dan 50%.  \n"
    "* Bij de titels hadden Masters (jongens), Miss's (meisjes) en Mrs's (vrouwen) een hoge overlevingskans. Mr's hadden een hele lage overlevingskans. \n"
    "* Bij de opstaplocaties was er een hoge overlevingskans bij Cherbourgh, en een lage bij Queenstown em Southampton.  \n"
    "* De 1e klas had nog redelijke overlevingskans. Daaropvolgde de 2e klas met een kans van 47%, dus meer overleefde het niet. De 3e klas had helaas geen geluk. \n"
    "* Als je alleen reisde was je overlevingskand klein, als je samenreisde was deze iets groter.")
    
    
    st.subheader('2.2 elkaar ondersteunende onafhankelijke variabelen tegenover overlevingskans')
    st.write('Naast de hierboven getrokken conclusies kan het zijn dat de overlevingskans hoger wordt als er combinaties gemaakt worden tussen de onafhanklijke variabelen.'
    'Om dit te bekijken kunnen 2 variabelen tegen overlevingskans geplot worden.')
    def plot_barchart(data, target='Survived'):
        # Keuze voor de onafhankelijke variabelen
        keuze = ['Type','Title','Age_categories','IsAlone','Embarked','Pclass',"Family", 'Small_fam']
        var1 = st.selectbox('Kies de eerste onafhankelijke variabele:', keuze)
        keuze.remove(var1)
        var2 = st.selectbox('Kies de tweede onafhankelijke variabele:', keuze)
        keuze.remove(var2)
        var3 = st.selectbox('Kies de derde onafhankelijke variabele:', keuze +['Geen filtering'])
        if var3 == 'Geen filtering':
            filtered_data = data
        else:
            var3_values = data[var3].unique()
            var3_value = st.selectbox(f'Kies de categorie voor {var3}:', var3_values)
            filtered_data = data[data[var3] == var3_value]

        # data filteren
        grouped_data = filtered_data[filtered_data[target] == 1].groupby([var1, var2]).size().reset_index(name='Aantal')
        total_data = data.groupby([var1, var2]).size().reset_index(name='Totaal')
        merged_data = pd.merge(grouped_data, total_data, on=[var1, var2])
        merged_data['Percentage'] = (merged_data['Aantal'] / merged_data['Totaal']) * 100

        # Maak de bar chart met percentages
        fig = px.bar(merged_data, x=var1, y='Percentage', color=var2,
                    labels={var1: var1, 'Percentage': f'%{target}', var2: var2},
                    title=f'Percentage {target} per {var1} en {var2}',
                    barmode='group')
        # fig.update_traces(text=merged_data['Percentage'].round(2).astype(str) + '%',
        #               textposition='outside',
        #               texttemplate='%{text}')
        fig.update_layout(yaxis=dict(range=[0, 100])) 

        # Toon de grafiek
        st.plotly_chart(fig)

    # Gebruik de functie
    plot_barchart(train_new)
    
    st.info("Intressante mogelijkheden zijn: \n" \
    "* Gelacht tegen leeftijd: \n"
    "   * Dit laat jonge mannen een hogere overlevingskans hebben. \n"
    "   * De vrouwen lijken allemaal een hoge overlevingskans te hebben \n"
    "   * Daarbij is wel goed zichtbaar dat als kinderen geen reisgezelschap hebben hun overlevingskans minimaal is. \n" 
    "   * Bekijk je de klasse erbij zie je dat vrouwen met een leeftijd tussen de 35 en 64 in de 1e klas een hogere overlevingskans hebben. \n"
    "* Title tegen opstaplocatie: \n"
    "   * Dit laat zien dat als Mrs's die op zijn gestapt in Queenstown geen grote overlevingskans hebben. \n"
    "- Uiteraard zijn er nog meer combinaties met informatie er uit te halen. Let hierbij wel goed op dat het soms eruit kan zien alof niemand het ergens overleefd heeft," \
    "Maar dit betekend waarschijnlijk dat er daar geen persoon van was (door de grootte van de dataset(klein)) "
    )
    
elif pagina == 'Voorspellend model':
    st.header('3. Voorspelling')
    train_new['Age'].fillna(train_new['Age'].median(), inplace=True)
    train_new['Embarked'].fillna('Southampthon', inplace=True)
    st.subheader('3.1 voorbereiding voorspelling')
    train_new['Embarked'] = train_new['Embarked'].map(embarked_mapping)
    train_new['Type'] = train_new['Type'].map(type_mapping)
    X = train_new.drop(columns=['Name', 'Sex', 'Parch', 'SibSp', 'Survived'], axis=1)
    y = train_new['Survived']
    st.write('De kolommen die meegenomen worden in de analyse zijn:')
    st.write(X.head())
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Modellen
    # Test verschillende waarden van n_neighbors
    accuracies = []
    neighbors_range = range(1, 4)

    for n in neighbors_range:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)


    st.subheader('3.2 KNN model')
    # Vind het beste aantal buren
    best_n = neighbors_range[accuracies.index(max(accuracies))]
    st.info(f"Beste aantal buren: {best_n}  \n"
            f"Hoogste nauwkeurigheid: {max(accuracies)}")
    # Plot de nauwkeurigheid voor verschillende waarden van n_neighbors
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(neighbors_range, accuracies, marker='o', linestyle='--', color='b')
    ax.set_title('Nauwkeurigheid vs. Aantal Buren (KNN)')
    ax.set_xlabel('Aantal Buren (n_neighbors)')
    ax.set_ylabel('Nauwkeurigheid')
    ax.set_xticks(neighbors_range)
    ax.grid()
    st.pyplot(fig)

    # Train een KNN-model met 3 buren
    knn_model = KNeighborsClassifier(n_neighbors=4)
    knn_scaler = StandardScaler()
    X_train_knn_scaled = pd.DataFrame(knn_scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_knn_scaled = pd.DataFrame(knn_scaler.transform(X_test), columns=X_test.columns)
    knn_model.fit(X_train_knn_scaled, y_train)
    knn_predictions = knn_model.predict(X_test_knn_scaled)

    # Bereken de nauwkeurigheid
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    st.write()

    test_df['Embarked'] = test_df['Embarked'].map(embarked_mapping)
    test_df['Type'] = test_df['Type'].map(type_mapping)
    test_df_knn = test_df.copy()

    # Zorg ervoor dat test_df dezelfde preprocessing heeft ondergaan als train_df
    X_test_df_knn = test_df_knn.drop(columns=['Name', 'Sex', 'Parch', 'SibSp', 'Survived'], errors='ignore')
    X_test_df_knn = X_test_df_knn.reindex(columns=X_train.columns, fill_value=0)

    # Schaal de testdata (gebruik dezelfde scaler als voor X_train)
    X_test_df_knn_scaled = pd.DataFrame(knn_scaler.transform(X_test_df_knn), columns=X_test_df_knn.columns)

    # Maak voorspellingen op test_df
    test_predictions_knn = knn_model.predict(X_test_df_knn_scaled)
    test_df_knn['Survived'] = test_predictions_knn

    # Sla de resultaten op in een CSV-bestand
    output_file_knn = "titanic_predictions_knn_3_neighbors.csv"
    test_df_knn[['PassengerId', 'Survived']].to_csv(output_file_knn, index=False)

    st.info(f"KNN Accuracy: {knn_accuracy}  \n"
            f"Voorspellingen opgeslagen in {output_file_knn}  \n"
            f"Aantal buren gebruikt in KNN-model: {knn_model.n_neighbors}")
    

    st.subheader('3.3 Lineare regressie')
    
    linear_model = LinearRegression()
    linear_scaler = StandardScaler()
    X_train_linear_scaled = pd.DataFrame(linear_scaler.fit_transform(X_train), columns=X_train.columns)  # Fit en transformeer de trainingsdata
    X_test_linear_scaled = pd.DataFrame(linear_scaler.transform(X_test), columns=X_test.columns)         # Transformeer de testdata
    linear_model.fit(X_train_linear_scaled, y_train)
    # Maak voorspellingen op de trainings-testset
    linear_predictions_raw = linear_model.predict(X_test_linear_scaled)

    # Afronden naar 0 of 1 voor classificatie
    linear_predictions = [1 if pred >= 0.88 else 0 for pred in linear_predictions_raw]

    # Evaluatie
    linear_accuracy = accuracy_score(y_test, linear_predictions)

    # Werk met een kopie van test_df
    test_df_linear = test_df.copy()

    # Zorg ervoor dat test_df dezelfde preprocessing heeft ondergaan als train_df
    X_test_df_linear = test_df_linear.drop(columns=['Name', 'Sex', 'Parch', 'SibSp', 'Survived'], errors='ignore')

    # Zorg ervoor dat de kolommen in test_df overeenkomen met die in X_train
    X_test_df_linear = X_test_df_linear.reindex(columns=X_train.columns, fill_value=0)

    # Schaal de testdata (gebruik dezelfde scaler als voor X_train)
    X_test_df_linear_scaled = pd.DataFrame(linear_scaler.transform(X_test_df_linear), columns=X_test_df_linear.columns)

    
    # Maak voorspellingen op de echte testdata
    test_predictions_linear_raw = linear_model.predict(X_test_df_linear_scaled)
    test_predictions_linear = [1 if pred >= 0.6 else 0 for pred in test_predictions_linear_raw]

    # Voeg de voorspellingen toe aan test_df
    test_df_linear['Survived'] = test_predictions_linear

    # Sla de resultaten op in een CSV-bestand
    output_file_linear = "titanic_lineaire_regressie_predictions.csv"
    test_df_linear[['PassengerId', 'Survived']].to_csv(output_file_linear, index=False)

    st.info(f"Accuracy (Lineaire Regressie): {linear_accuracy}   \n"
          f"Voorspellingen opgeslagen in {output_file_linear}")
    
    st.subheader('3.4 logistische regressie')
    # Train een logistisch regressiemodel
    logistic_model = LogisticRegression(max_iter=1000)

    # Zorg ervoor dat de scaler wordt getraind op de trainingsdata
    logistic_scaler = StandardScaler()
    X_train_logistic_scaled = pd.DataFrame(logistic_scaler.fit_transform(X_train), columns=X_train.columns)  # Fit en transformeer de trainingsdata
    X_test_logistic_scaled = pd.DataFrame(logistic_scaler.transform(X_test), columns=X_test.columns)         # Transformeer de testdata

    # Train het logistische regressiemodel
    logistic_model.fit(X_train_logistic_scaled, y_train)

    # Maak voorspellingen op de trainings-testset
    logistic_predictions = logistic_model.predict(X_test_logistic_scaled)

    # Evaluatie
    logistic_accuracy = accuracy_score(y_test, logistic_predictions)

    # Werk met een kopie van test_df
    test_df_logistic = test_df.copy()

    # Zorg ervoor dat test_df dezelfde preprocessing heeft ondergaan als train_df
    X_test_df_logistic = test_df_logistic.drop(columns=['Title', 'Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Parch', 'SibSp'], axis=1)

    # Controleer of er ontbrekende waarden zijn in X_test_df_logistic en vul deze alleen voor numerieke kolommen
    numeric_cols_logistic = X_test_df_logistic.select_dtypes(include=['float64', 'int64', 'int32', 'int16'])
    X_test_df_logistic[numeric_cols_logistic.columns] = numeric_cols_logistic.fillna(numeric_cols_logistic.mean())

    # Zorg ervoor dat de kolommen in test_df overeenkomen met die in X_train
    X_test_df_logistic = X_test_df_logistic.reindex(columns=X_train.columns, fill_value=0)

    # Schaal de testdata (gebruik dezelfde scaler als voor X_train)
    X_test_df_logistic_scaled = pd.DataFrame(logistic_scaler.transform(X_test_df_logistic), columns=X_test_df_logistic.columns)

    # Maak voorspellingen op test_df
    test_predictions_logistic = logistic_model.predict(X_test_df_logistic_scaled)

    # Voeg de voorspellingen toe aan test_df
    test_df_logistic['Survived'] = test_predictions_logistic

    # Sla de resultaten op in een CSV-bestand
    output_file_logistic = "titanic_logistische_regressie_predictions.csv"
    test_df_logistic[['PassengerId', 'Survived']].to_csv(output_file_logistic, index=False)

    st.info(f"Accuracy (Logistische Regressie): {logistic_accuracy}  \n"
            f"Voorspellingen opgeslagen in {output_file_logistic}")