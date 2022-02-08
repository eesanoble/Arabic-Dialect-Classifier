import streamlit as st

#Headings for Web Application
st.title("The Humble Arabic Dialect Classifier")

st.markdown("""The Humble Arabic Dialect Classifier is trained on a selection of
tweets written in dialectical Arabic. Based on the vocabulary and morphology of a
particular text, the HADC can ascertain which region a text is likely to have
come from.""")

st.subheader("Enter a text written in dialectical Arabic.")
text = st.text_input('Enter text') #text is stored in this variable

st.subheader("Results")
