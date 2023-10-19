import langchain_helper as lch
import streamlit as st

st.title("Pets name generator")


user_animal_type = st.sidebar.selectbox("What is your pet?",("Cat","Dog","Hamster","Rabbit"))


user_pet_color = st.sidebar.text_area(label=f"What color is your {user_animal_type}?",max_chars=10).strip()



if user_pet_color:
    response = lch.generate_pet_name(user_animal_type,user_pet_color)
    st.text(response['pet_names'])
