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
'De 1e keer met 2 weken data science kennis en de 2e keer met 9 weken kennis.'
'  \nMet de 2e keer, zijn de visualisaties verbeterd, zijn nieuwe variabele opgeroepen en is er statistisch meer naar de modellen gekeken.')

st.header('Opzet')
st.write('Voor de opzet is gekozen om gebruik te maken van de streamlit multipager. Dit zorgt ervoor dat je met meerdere streamlit py files werkt en 1 file.'
'   \nIn de sidebar valt de keuze te maken tussen de versies (files). Iedere versie is onderverdeeld in hoofdstukken (te kiezen via de inhoudsopgave) en paragraven (in de tekst beschreven)')

st.header('Veranderingen')
st.write('Zoals al aangegeven zijn er verschillende veranderingen gedaan. '
'Om deze veranderingen inzichtelijk te maken is er een keuze te maken voor de soort verandering. De verandering valt te kieze in de sidebar (onder inhoudsopgave).')

with st.sidebar: 
    pagina = option_menu('Inhoudsopgave', ["Nieuwe variabelen", "Visualisaties", "Statistiek en modellen"], icons=['search', 'bar-chart-line', 'graph-up-arrow'], menu_icon='card-list')


if pagina == "Nieuwe variabelen":
    st.subheader(pagina)
    st.write(""
    "- **Bij de eerste keer** is de kolom **Leeftijdscategorie** toegevoegd.  \n"
    "- **Bij de tweede keer** zijn naast de kolom **Leeftijdscategorie** ook de kolommen:  \n"
    "   - **Titel**   \n"
    "   - **Alleen reizen**  \n" \
    "   - **Familie**  \n"
    "   - **Familie grootte**  \n" \
    "   - **Type (man/rouw/kind)**")

elif pagina == "Visualisaties":
    st.subheader(pagina)
    st.write(""
    "- **Eerste poging:**  \n"
    "   - De **correlatiematrix** kreeg feedback dat deze mooier kon.  \n"
     "  - De eerste keer waren de figuren gemaakt met **Matplotlib** en soms **Seaborn**.  \n"
    "- **Tweede poging:**  \n"
    "   - De correlatiematrix is verbeterd: van een **getalmatig overzicht** naar een **figuur met kleur en getallen**.  \n"
    "   - De figuren zijn grotendeels gemaakt met **Plotly Express**, en sommige nog met **Seaborn** en nog een enkele met **Matplotlib**.  \n")

elif pagina == "Statistiek en modellen":
    st.subheader(pagina)
    st.write(
        "Bij de eerste keer is er niet direct een algoritme gebruikt. Door middel van figuren en percentages zijn er keuzes gemaakt. "
        "Hieruit zijn voorwaarden gesteld om het resultaat te bepalen. Met deze methode is er een Kaggle-score gehaald van **78,47%**.\n"
        
        "\nBij de tweede keer zijn verschillende algoritmen gebruikt. Er is gekozen om gebruik te maken van **KNN**, **logistische** en **lineaire regressie**. "
        "Hiermee werden scores gehaald van **70,81%**, **76,08%** en **77,75%**.  \n" \
        
        "\nUiteindelijk is er dus niet hoger gehaald met de algoritme en waren de gemaakte keuzes beter voor de accuraatheid."
    )

    