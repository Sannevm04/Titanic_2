import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Titanic verbetercase",
    page_icon="❄️🚢",
    layout="wide"
)

st.title("Veranderingen op de Titanic Case")
st.caption('Team 5, Quinn en Sanne')
st.write('Welkom op deze site, waar de Titanic case 2 keer is uitgevoerd. '
'De eerste keer met 2 weken programmeer kennis en de tweede keer met 9 weken programmeer kennis.'
'  \nMet de 2e keer, zijn de visualisaties verbeterd, zijn nieuwe variabele opgeroepen en is er statistisch meer naar de modellen gekeken.')

st.markdown('# Opzet')
st.write('Voor de opzet is gekozen om gebruik te maken van de streamlit multipager. Dit zorgt ervoor dat je met meerdere streamlit py files werkt en 1 hoofddocument.'
'   \nIn de sidebar valt te kiezen welke webpagina gebruikt wordt(welk thema), dan zijn er nog hoofdstukken en paragraven in de tekst beschreven')

st.markdown('# Veranderingen')
st.write('Zoals al aangegeven zijn er verschillende veranderingen gedaan. '
'Om deze veranderingen inzichtelijk te maken is er een keuze te maken voor de soort verandering:')

st.title("🚢 Verbeterpoging")
with st.sidebar: 
    pagina = option_menu('Inhoudsopgave', ["Nieuwe variabelen", "Visualisaties", "Statistiek en modellen"], icons=['search', 'bar-chart-line', 'graph-up-arrow'], menu_icon='card-list')


if pagina == "Nieuwe variabelen":
    st.header(pagina)
    st.write(""
    "- **Bij de eerste keer** is de kolom **Leeftijdscategorie** toegevoegd.  \n"
    "- **Bij de tweede keer** zijn naast de kolom **Leeftijdscategorie** ook de kolommen:  \n"
    "   - **Titel**   \n"
    "   - **Reisgenoten**    \n"
    "   - **Alleen reizen**")

elif pagina == "Visualisaties":
    st.header(pagina)
    st.write(""
    "- **Eerste keer:** De **correlatiematrix** kreeg feedback dat deze mooier kon.  \n"
    "- **Tweede poging:**  \n"
    "   - De correlatiematrix is verbeterd: van een **getalmatig overzicht** naar een **figuur met kleur en getallen**.  \n"
    "   - De eerste keer waren de figuren gemaakt met **Matplotlib** en soms **Seaborn**.  \n"
    "   - Nu zijn de figuren grotendeels gemaakt met **Plotly Express**, en sommige nog met **Seaborn**.  \n")

elif pagina == "Statistiek en modellen":
    st.header(pagina)
    st.write(
        "Bij de eerste keer is er niet direct een algoritme gebruikt. Door middel van figuren en percentages zijn er keuzes gemaakt. "
        "Hieruit zijn voorwaarden gesteld om het resultaat te bepalen. Met deze methode is er een Kaggle-score gehaald van **78,47%**.\n"
        
        "\nBij de tweede keer zijn verschillende algoritmen gebruikt. Er is gekozen om gebruik te maken van **KNN**, **logistische** en **lineaire regressie**. "
        "Hiermee werden scores gehaald van **70,81% **76,08%** en **77,75%**."
    )

    